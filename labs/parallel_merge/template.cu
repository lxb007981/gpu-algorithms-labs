#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 128
#define TILE_SIZE 512

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
  return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/

__device__ int co_rank(float* A, int A_len, float* B, int B_len, int k) {
  /* Your code here */
  int i_min = max(0, k - B_len);
  int i_max = min(A_len, k);
  while (i_min < i_max) {
    int i = i_min + (i_max - i_min) / 2;
    int j = k - i;
    if (i > 0 && j < B_len && A[i - 1] > B[j])
      i_max = i - 1;
    else if (j > 0 && i < A_len && A[i] <= B[j - 1])
      i_min = i + 1;
    else
      return i;
  }
  return i_min;
}

__device__ int co_rank_circular(int k, float* A, int A_len, float* B, int B_len, int A_S_start, int B_S_start) {
  int i     = min(A_len, k);
  int j     = k - i;
  int i_low = max(0, k - B_len);
  int j_low = max(0, k - A_len);
  int delta;
  while (true) {
    int i_cir         = (i + A_S_start) % TILE_SIZE;
    int i_minus_1_cir = (i + A_S_start - 1) % TILE_SIZE;
    int j_cir         = (j + B_S_start) % TILE_SIZE;
    int j_minus_1_cir = (j + B_S_start - 1) % TILE_SIZE;
    if (i > 0 && j < B_len && A[i_minus_1_cir] > B[j_cir]) {
      delta = ceil_div(i - i_low, 2);
      j_low = j;
      i -= delta;
      j += delta;
    } else if (j > 0 && i < A_len && A[i_cir] <= B[j_minus_1_cir]) {
      delta = ceil_div(j - j_low, 2);
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      return i;
    }
  }
}
/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
  int i = 0, j = 0, k = 0;

  while ((i < A_len) && (j < B_len)) {
    C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
  }

  if (i == A_len) {
    while (j < B_len) {
      C[k++] = B[j++];
    }
  } else {
    while (i < A_len) {
      C[k++] = A[i++];
    }
  }
}

__device__ void merge_sequential_circular(float* A, int A_len, float* B, int B_len, float* C, int A_S_start, int B_S_start) {
  int i = 0, j = 0, k = 0;

  while ((i < A_len) && (j < B_len)) {
    int i_cir = (i + A_S_start) % TILE_SIZE;
    int j_cir = (j + B_S_start) % TILE_SIZE;
    if (A[i_cir] <= B[j_cir]) {
      C[k++] = A[i_cir];
      i++;
    } else {
      C[k++] = B[j_cir];
      j++;
    }
  }

  if (i == A_len) { // A is exhausted
    while (j < B_len) {
      int j_cir = (j + B_S_start) % TILE_SIZE;
      C[k++]    = B[j_cir];
      j++;
    }
  } else { // B is exhausted
    while (i < A_len) {
      int i_cir = (i + A_S_start) % TILE_SIZE;
      C[k++]    = A[i_cir];
      i++;
    }
  }
}
/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
  /* Your code here */
  int tid            = threadIdx.x + blockIdx.x * blockDim.x;
  int elt_per_thread = ceil_div(A_len + B_len, blockDim.x * gridDim.x);
  int start          = min(tid * elt_per_thread, A_len + B_len);
  int end            = min((tid + 1) * elt_per_thread, A_len + B_len);
  int i_start        = co_rank(A, A_len, B, B_len, start);
  int j_start        = start - i_start;
  int i_end          = co_rank(A, A_len, B, B_len, end);
  int j_end          = end - i_end;
  merge_sequential(A + i_start, i_end - i_start, B + j_start, j_end - j_start, C + start);
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
  /* Your code here */
  __shared__ float shareAB[TILE_SIZE * 2];
  float* shareA     = &shareAB[0];
  float* shareB     = &shareAB[TILE_SIZE];
  int elt_per_block = ceil_div(A_len + B_len, gridDim.x);
  int blk_C_start   = blockIdx.x * elt_per_block;
  int blk_C_end     = min((blockIdx.x + 1) * elt_per_block, A_len + B_len);
  if (threadIdx.x == 0) {
    shareA[0] = co_rank(A, A_len, B, B_len, blk_C_start);
    shareA[1] = co_rank(A, A_len, B, B_len, blk_C_end);
  }
  __syncthreads();
  int blk_A_start = shareA[0];
  int blk_A_end   = shareA[1];
  int blk_B_start = blk_C_start - blk_A_start;
  int blk_B_end   = blk_C_end - blk_A_end;
  __syncthreads();
  int blk_C_len = blk_C_end - blk_C_start;
  int blk_A_len = blk_A_end - blk_A_start;
  int blk_B_len = blk_B_end - blk_B_start;

  int num_tile = ceil_div(blk_C_len, TILE_SIZE);

  int C_produced = 0;
  int A_consumed = 0;
  int B_consumed = 0;

  for (int counter = 0; counter < num_tile; counter++) {
    // load one tile
    for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
      if (i + threadIdx.x + A_consumed < blk_A_len) {
        shareA[i + threadIdx.x] = A[blk_A_start + i + threadIdx.x + A_consumed];
      }
      if (i + threadIdx.x + B_consumed < blk_B_len) {
        shareB[i + threadIdx.x] = B[blk_B_start + i + threadIdx.x + B_consumed];
      }
    }
    __syncthreads();
    // merge one tile
    int per_thread  = (TILE_SIZE/ blockDim.x);
    int C_remain    = blk_C_len - C_produced;
    int thr_C_start = min(threadIdx.x * per_thread, C_remain);
    int thr_C_end   = min((threadIdx.x + 1) * per_thread, C_remain);

    int A_remain = min(blk_A_len - A_consumed, TILE_SIZE);
    int B_remain = min(blk_B_len - B_consumed, TILE_SIZE);

    int thr_A_start = co_rank(shareA, A_remain, shareB, B_remain, thr_C_start);
    int thr_B_start = thr_C_start - thr_A_start;
    int thr_A_end   = co_rank(shareA, A_remain, shareB, B_remain, thr_C_end);
    int thr_B_end   = thr_C_end - thr_A_end;

    merge_sequential(shareA + thr_A_start, thr_A_end - thr_A_start, shareB + thr_B_start, thr_B_end - thr_B_start,
                     C + blk_C_start + C_produced + thr_C_start);

    C_produced += TILE_SIZE;
    A_consumed += co_rank(shareA, TILE_SIZE, shareB, TILE_SIZE, TILE_SIZE);
    B_consumed = C_produced - A_consumed;
    __syncthreads();
  }
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    __shared__ float shareAB[TILE_SIZE * 2];
    float* shareA     = &shareAB[0];
    float* shareB     = &shareAB[TILE_SIZE];
    int elt_per_block = ceil_div(A_len + B_len, gridDim.x);
    int blk_C_start   = blockIdx.x * elt_per_block;
    int blk_C_end     = min((blockIdx.x + 1) * elt_per_block, A_len + B_len);
    if (threadIdx.x == 0) {
      shareA[0] = co_rank(A, A_len, B, B_len, blk_C_start);
      shareA[1] = co_rank(A, A_len, B, B_len, blk_C_end);
    }
    __syncthreads();
    int blk_A_start = shareA[0];
    int blk_A_end   = shareA[1];
    int blk_B_start = blk_C_start - blk_A_start;
    int blk_B_end   = blk_C_end - blk_A_end;
    __syncthreads();

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = TILE_SIZE; // the number of elements that need to be filled up in the next iteration
    int B_S_consumed = TILE_SIZE; // in the first iteration, we need to fill up the whole buffer

    int blk_C_len = blk_C_end - blk_C_start;
    int blk_A_len = blk_A_end - blk_A_start;
    int blk_B_len = blk_B_end - blk_B_start;

    int num_tile = ceil_div(blk_C_len, TILE_SIZE);

    int C_produced = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    for (int counter = 0; counter < num_tile; counter++) {
        for(int i = 0; i < A_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x + A_consumed < blk_A_len && i + threadIdx.x < A_S_consumed) {
                shareA[(A_S_start + (TILE_SIZE - A_S_consumed) + i + threadIdx.x) % TILE_SIZE] = A[blk_A_start + A_consumed + i + threadIdx.x];
            }
        }
        for(int i = 0; i < B_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x + B_consumed < blk_B_len && i + threadIdx.x < B_S_consumed) {
                shareB[(B_S_start + (TILE_SIZE - B_S_consumed) + i + threadIdx.x) % TILE_SIZE] = B[blk_B_start + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();
        int per_thread  = TILE_SIZE / blockDim.x;
        int thr_C_start = min(threadIdx.x * per_thread, blk_C_len - C_produced);
        int thr_C_end   = min((threadIdx.x + 1) * per_thread, blk_C_len - C_produced);

        int A_remain = min(blk_A_len - A_consumed, TILE_SIZE);
        int B_remain = min(blk_B_len - B_consumed, TILE_SIZE);

        int thr_A_start = co_rank_circular(thr_C_start, shareA, A_remain, shareB, B_remain, A_S_start, B_S_start);
        int thr_B_start = thr_C_start - thr_A_start;
        int thr_A_end   = co_rank_circular(thr_C_end, shareA, A_remain, shareB, B_remain, A_S_start, B_S_start);
        int thr_B_end   = thr_C_end - thr_A_end;

        merge_sequential_circular(shareA, thr_A_end - thr_A_start, shareB, thr_B_end - thr_B_start, C + blk_C_start + C_produced + thr_C_start, A_S_start + thr_A_start, B_S_start + thr_B_start);

        A_S_consumed = co_rank_circular(min(TILE_SIZE, blk_C_len - C_produced), shareA, A_remain, shareB, B_remain, A_S_start, B_S_start);
        B_S_consumed = min(TILE_SIZE, blk_C_len - C_produced) - A_S_consumed;
        A_consumed += A_S_consumed;
        C_produced += min(TILE_SIZE, blk_C_len - C_produced);
        B_consumed = C_produced - A_consumed;

        A_S_start = (A_S_start + A_S_consumed) % TILE_SIZE;
        B_S_start = (B_S_start + B_S_consumed) % TILE_SIZE;
        __syncthreads();
    }
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
  const int numBlocks = 128;
  gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
  const int numBlocks = 128;
  gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
  const int numBlocks = 128;
  gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
