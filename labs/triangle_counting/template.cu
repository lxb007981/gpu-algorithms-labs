#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define COARSE_FACTOR 2
#define BLOCK_DIM 512

__device__ uint64_t linear_search(const uint32_t *const edgeDst, uint32_t start1, uint32_t end1, uint32_t start2, uint32_t end2) {
  uint64_t count = 0;
  while (true) {
    if (start1 == end1 || start2 == end2) {
      break;
    }
    if (edgeDst[start1] == edgeDst[start2]) {
      count++;
      start1++;
      start2++;
    } else if (edgeDst[start1] < edgeDst[start2]) {
      start1++;
    } else {
      start2++;
    }
  }
  return count;
}

__device__ bool binary_search(const uint32_t *const edgeDst, uint32_t start, uint32_t end, uint32_t target) {
  // search includes start and end
  while (start <= end) {
    uint32_t mid = start + (end - start) / 2;
    if (edgeDst[mid] == target) {
      return true;
    } else if (edgeDst[mid] < target) {
      start = mid + 1;
    } else {
      end = mid - 1;
    }
  }
  return false;
}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge
  uint32_t edgeId = blockIdx.x * blockDim.x + threadIdx.x;
  if (edgeId >= numEdges) {
    return;
  }
  uint32_t src = edgeSrc[edgeId];
  uint32_t dst = edgeDst[edgeId];
  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  uint32_t start1 = rowPtr[src];
  uint32_t end1   = rowPtr[src + 1];
  uint32_t start2 = rowPtr[dst];
  uint32_t end2   = rowPtr[dst + 1];
  // Determine how many elements of those two arrays are common
  uint64_t count = linear_search(edgeDst, start1, end1, start2, end2);
  // Store the result in the triangleCounts array
  triangleCounts[edgeId] = count;
}

__global__ static void kernel_tc_binary(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                        const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                        const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                        const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                        const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge
  uint32_t edgeId = blockIdx.x * blockDim.x + threadIdx.x;
  if (edgeId >= numEdges) {
    return;
  }
  uint32_t src = edgeSrc[edgeId];
  uint32_t dst = edgeDst[edgeId];
  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  uint32_t start1 = rowPtr[src];
  uint32_t end1   = rowPtr[src + 1];
  uint32_t start2 = rowPtr[dst];
  uint32_t end2   = rowPtr[dst + 1];
  if (end1 - start1 > end2 - start2) {
    uint32_t temp = start1;
    start1        = start2;
    start2        = temp;
    temp          = end1;
    end1          = end2;
    end2          = temp;
  }
  // now end1 - start1 <= end2 - start2
  // Determine how many elements of those two arrays are common
  uint64_t count = 0;
  for (uint32_t i = start1; i < end1; i++) {
    if (binary_search(edgeDst, start2, end2 - 1, edgeDst[i])) {
      count++;
    }
  }
  // Store the result in the triangleCounts array
  triangleCounts[edgeId] = count;
}

__global__ static void kernel_tc_mixed(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                       const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                       const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                       const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                       const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge
  uint32_t edgeId = blockIdx.x * blockDim.x + threadIdx.x;
  if (edgeId >= numEdges) {
    return;
  }
  uint32_t src = edgeSrc[edgeId];
  uint32_t dst = edgeDst[edgeId];
  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  uint32_t start1 = rowPtr[src];
  uint32_t end1   = rowPtr[src + 1];
  uint32_t start2 = rowPtr[dst];
  uint32_t end2   = rowPtr[dst + 1];
  if (end1 - start1 > end2 - start2) {
    uint32_t temp = start1;
    start1        = start2;
    start2        = temp;
    temp          = end1;
    end1          = end2;
    end2          = temp;
  }
  // now end1 - start1 <= end2 - start2
  uint32_t U     = end1 - start1;
  uint32_t V     = end2 - start2;
  uint64_t count = 0;
  if (V > 64 && V / U > 6) { // Determine how many elements of those two arrays are common
    for (uint32_t i = start1; i < end1; i++) {
      if (binary_search(edgeDst, start2, end2 - 1, edgeDst[i])) {
        count++;
      }
    }
  } else {
    count = linear_search(edgeDst, start1, end1, start2, end2);
  }
  // Store the result in the triangleCounts array
  triangleCounts[edgeId] = count;
}

__global__ void CoarsenedSumReductionKernel(const uint64_t *const triangleCounts, //!< per-edge triangle counts
                                            uint32_t *const total,                //!< total number of triangles
                                            const size_t numEdges                 //!< how many edges to count triangles for
) {
  __shared__ uint32_t input_s[BLOCK_DIM];
  uint32_t segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  uint32_t i       = segment + threadIdx.x;
  uint32_t t       = threadIdx.x;
  input_s[t]       = 0;
  if (i >= numEdges) {
    return;
  }
  uint32_t sum = triangleCounts[i];
  for (uint32_t tile = 1; tile < COARSE_FACTOR * 2; tile++) {
    if (i + tile * BLOCK_DIM < numEdges) {
      sum += triangleCounts[i + tile * BLOCK_DIM];
    } else {
      break;
    }
  }
  input_s[t] = sum;
  for (uint32_t stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (t == 0) {
    atomicAdd(total, input_s[0]);
  }
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
  const size_t numEdges = view.nnz();
  pangolin::Vector<uint64_t> triangleCounts(numEdges, 0);
  uint32_t total = 0;
  uint32_t *d_total;
  cudaMalloc(&d_total, sizeof(uint32_t));
  cudaMemcpy(d_total, &total, sizeof(uint32_t), cudaMemcpyHostToDevice);

  dim3 dimBlock(512);
  //@@ calculate the number of blocks needed
  // dim3 dimGrid (ceil(number of non-zeros / dimBlock.x))
  dim3 dimGrid(ceil(numEdges / (double)dimBlock.x));

  if (mode == 1) {

    //@@ launch the linear search kernel here
    // kernel_tc<<<dimGrid, dimBlock>>>(...)
    kernel_tc<<<dimGrid, dimBlock>>>(triangleCounts.data(), view.row_ind(), view.col_ind(), view.row_ptr(), numEdges);
  } else if (2 == mode) {

    //@@ launch the hybrid search kernel here
    // your_kernel_name_goes_here<<<dimGrid, dimBlock>>>(...)
    kernel_tc_binary<<<dimGrid, dimBlock>>>(triangleCounts.data(), view.row_ind(), view.col_ind(), view.row_ptr(), numEdges);
  } else if (3 == mode) {

    //@@ launch the hybrid search kernel here
    // your_kernel_name_goes_here<<<dimGrid, dimBlock>>>(...)
    kernel_tc_mixed<<<dimGrid, dimBlock>>>(triangleCounts.data(), view.row_ind(), view.col_ind(), view.row_ptr(), numEdges);
  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }

  //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
  cudaDeviceSynchronize();
  /*
  for (uint32_t i = 0; i < numEdges; i++) {
    total += triangleCounts[i];
  }
  */
  dim3 dimGrid2((numEdges + COARSE_FACTOR * 2 * BLOCK_DIM - 1) / (COARSE_FACTOR * 2 * BLOCK_DIM));
  dim3 dimBlock2(BLOCK_DIM);
  CoarsenedSumReductionKernel<<<dimGrid2, dimBlock2>>>(triangleCounts.data(), d_total, numEdges);
  cudaDeviceSynchronize();
  cudaMemcpy(&total, d_total, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  return total;
}
