#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <vector>

#define COARSE_FACTOR 2
#define BLOCK_DIM 512

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

int main() {
  // test reduction
  const size_t numEdges = 12134;
  std::vector<uint64_t> triangleCounts(numEdges);
  for (size_t i = 0; i < numEdges; i++) {
    // assign random integer between 0 and 1
    triangleCounts[i] = 2;
  }
  uint64_t *d_triangleCounts;
  uint32_t *d_total;
  cudaMalloc(&d_triangleCounts, numEdges * sizeof(uint64_t));
  cudaMalloc(&d_total, sizeof(uint32_t));
  cudaMemcpy(d_triangleCounts, triangleCounts.data(), numEdges * sizeof(uint64_t), cudaMemcpyHostToDevice);
  uint32_t total = 0;
  cudaMemcpy(d_total, &total, sizeof(uint32_t), cudaMemcpyHostToDevice);
  dim3 block(BLOCK_DIM);
  dim3 grid((numEdges + COARSE_FACTOR * 2 * BLOCK_DIM - 1) / (COARSE_FACTOR * 2 * BLOCK_DIM));
  CoarsenedSumReductionKernel<<<grid, block>>>(d_triangleCounts, d_total, numEdges);
  cudaDeviceSynchronize();
  cudaMemcpy(&total, d_total, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  printf("total: %u\n", total);
}