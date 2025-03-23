#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128   // T
#define TILE_SZ_B 16    // U
#define TILE_SZ_RATIO 8 // (TILE_SZ_A / TILE_SZ_B) // i.e. S
__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C) {

/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is row major
 *
 ********************************************************************/

// Macros for accessing flattened matrices
#define A(row, col) A[(row) + (col) *m]
#define B(row, col) B[(row) *n + (col)]
#define C(row, col) C[(row) + (col) *m]

  // INSERT KERNEL CODE HERE

  // SSL Hint (9/6/21): try using just one register for the tile of A
  // rather than several--in other words, load one value (per thread)
  // from A and compute using that value rather than loading all values
  // before doing the computation.  This approach seems to be slightly
  // faster than the alternative.

  // collaborately load SxU tile of B into shared memory
  __shared__ float B_tile[TILE_SZ_RATIO][TILE_SZ_B];
  int b_col_start = blockIdx.x * TILE_SZ_B;
  int b_col_end   = b_col_start + TILE_SZ_B;
  int b_x         = threadIdx.y % TILE_SZ_B;
  int b_y         = threadIdx.y / TILE_SZ_B;
  int b_col       = b_x + b_col_start;
  int a_row       = threadIdx.y + blockIdx.y * TILE_SZ_A;
  int b_row       = 0;
  for (int b_col = b_col_start; b_col < b_col_end; b_col++) {
    if (a_row < m && b_col < n) {
      C(a_row, b_col) = 0;
    }
  }

  for (int i = 0; i < k; i += TILE_SZ_RATIO) {
    b_row = b_y + i;
    if (b_row < k && b_col < n) {
      B_tile[b_y][b_x] = B(b_row, b_col);
    }

    __syncthreads();
    // load one value from A into register and start computing
    for (int a_col = i; a_col < i + TILE_SZ_RATIO; a_col++) {
      if (a_row >= m || a_col >= k) {
        continue;
      }
      float a_val = A(a_row, a_col);
      for (int b_col = b_col_start; b_col < b_col_end; b_col++) {
        if (a_row < m && b_col < n) {
          C(a_row, b_col) += a_val * B_tile[a_col - i][b_col - b_col_start];
        }
      }
    }
    __syncthreads();
  }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                float *C, int ldc) {
  if ((transa != 'N') && (transa != 'n')) {
    printf("unsupported value of 'transa'\n");
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    printf("unsupported value of 'transb'\n");
    return;
  }

  if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
    printf("unsupported value of alpha\n");
    return;
  }

  if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
    printf("unsupported value of beta\n");
    return;
  }

  // Initialize thread block and kernel grid dimensions ---------------------

  // Your code need only consider the m, n, k, A, B, and C parameters of
  // the function, which provide the matrix sizes (m, n, k) and data
  // (A, B, C).

  // INSERT CODE HERE
  dim3 dimBlock(1, TILE_SZ_A, 1);
  dim3 dimGrid((n + TILE_SZ_B - 1) / TILE_SZ_B, (m + TILE_SZ_A - 1) / TILE_SZ_A, 1);
  mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
  // Invoke CUDA kernel -----------------------------------------------------

  // INSERT CODE HERE
}
