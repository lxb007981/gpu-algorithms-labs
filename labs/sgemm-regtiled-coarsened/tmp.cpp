#include <random>
#include <iostream>
#include <stdio.h>
using namespace std;
#define TILE_SZ_A 128                         // T
#define TILE_SZ_B 16                          // U
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B) // S
#define A(row, col) hostA[(row) + (col) *m]
#define B(row, col) hostB[(row) *n + (col)]
#define C(row, col) hostC[(row) + (col) *m]

static uint_fast32_t *rng_new_state(uint_fast32_t seed) {
  uint64_t *rng_state = new uint64_t;
  *rng_state          = seed;
  return rng_state;
}

static uint_fast32_t *rng_new_state() {
  return rng_new_state(88172645463325252LL);
}

static uint_fast32_t rng_uint32(uint_fast32_t *rng_state) {
  uint_fast32_t local = *rng_state;
  local ^= local << 13; // a
  local ^= local >> 17; // b
  local ^= local << 5;  // c
  *rng_state = local;
  return local;
}

static float rng_float(uint_fast32_t *state) {
  uint_fast32_t rnd = rng_uint32(state);
  const auto r      = static_cast<float>(rnd) / static_cast<float>(UINT_FAST32_MAX);
  if (std::isfinite(r)) {
    return r;
  }
  return rng_float(state);
}

static void generate_data(float *x, const size_t n) {
  const auto rng_state = rng_new_state();

  for (size_t ii = 0; ii < n; ++ii) {
    x[ii] = rng_float(rng_state);
  }

  delete rng_state;
}
void verify(const float *A, const float *B, const float *C, size_t m, size_t k, size_t n) {

  for (size_t row = 0; row < m; ++row) {
    for (size_t col = 0; col < n; ++col) {
      float sum = 0;
      for (size_t i = 0; i < k; ++i) {
        sum += A[row + i * m] * B[i * n + col];
      }
      float relativeError = (sum - C[row + col * m]) / sum;
      if (std::abs(relativeError) >= 1e-6) {
        std::cerr << "the results were not close enough at C[" << row << "," << col << "], expected " << sum << " got " << C[row + col * m]
                  << std::endl;
        exit(-1);
      }
      else{
        std::cout << "the results were close enough at C[" << row << "," << col << "], expected " << sum << " got " << C[row + col * m]
                  << std::endl;
      }
    }
  }
}
int main() {
  int m = 32, k = 32, n = 32;
  int blockIdxX = 0, blockIdxY = 0;
  int threadIdxX = 0, threadIdxY = 0;
  float B_tile[TILE_SZ_RATIO][TILE_SZ_B];
  int b_col_start = blockIdxX * TILE_SZ_B;
  int b_col_end   = b_col_start + TILE_SZ_B;
  // int b_row_start       = threadIdxY / TILE_SZ_B ;
  int b_x         = threadIdxY % TILE_SZ_B;
  int b_y         = threadIdxY / TILE_SZ_B;
  int b_col       = b_x + b_col_start;
  int a_row       = threadIdxY + blockIdxY * TILE_SZ_A;
  int b_row       = b_y - TILE_SZ_RATIO;
  int a_col_start = -TILE_SZ_RATIO;
  int a_col_end   = a_col_start + TILE_SZ_RATIO;
  int iDebug      = 0;

  const size_t aSz = m * k;
  const size_t bSz = k * n;
  const size_t cSz = m * n;

  // generate input data
  std::vector<float> hostA(aSz);
  std::vector<float> hostB(bSz);
  std::vector<float> hostC(cSz);
  generate_data(hostA.data(), hostA.size());
  generate_data(hostB.data(), hostB.size());

  for (int i = 0; i < k; i += TILE_SZ_RATIO) {
    b_row += TILE_SZ_RATIO;
    a_col_start += TILE_SZ_RATIO;
    a_col_end += TILE_SZ_RATIO;
    for (int threadIdxY = 0; threadIdxY < TILE_SZ_A; threadIdxY++) {
      b_y              = threadIdxY / TILE_SZ_B;
      b_x              = threadIdxY % TILE_SZ_B;
      B_tile[b_y][b_x] = B(i  + b_y,  b_x);
    }
    if (b_row < k && b_col < n) {
      // if (i == iDebug ) {
      //   printf("b_row: %d, b_col: %d ", b_row, b_col);
      //   printf("b_y: %d, b_x: %d\n", b_y, b_x);
      // }
      // B_tile[b_y][b_x] = B(b_row, b_col);
    }
    // load one value from A into register and start computing
    for (int a_col = a_col_start; a_col < a_col_end; a_col++) {
      float a_val = A(a_row, a_col);
      // if (i == iDebug) {
      // printf("a_row: %d, a_col: %d\n\n", a_row, a_col);
      // }
      for (int b_col = b_col_start; b_col < b_col_end; b_col++) {
        // if (i == iDebug) {
        //   printf("a_row:\t\t%d, b_col:\t%d\n", a_row, b_col);
        //  printf("b_tile_y:\t%d, b_tile_x:\t%d\n", a_col - a_col_start, b_col - b_col_start);
        // }
        C(a_row, b_col) += a_val * B_tile[a_col - a_col_start][b_col - b_col_start];
      }
    }
  }
  verify(hostA.data(), hostB.data(), hostC.data(), 32, 32, 32);
}