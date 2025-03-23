#include <cstdio>
#include <cstdlib>

#include "assert.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include <cuda.h>

#define TILE_SIZE 30

__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {

// INSERT KERNEL CODE HERE
#define A0(i, j, k) A0[((k) * ny + (j)) * nx + (i)]
#define Anext(i, j, k) Anext[((k) * ny + (j)) * nx + (i)]
  int i = blockIdx.x * TILE_SIZE + threadIdx.x;
  int j = blockIdx.y * TILE_SIZE + threadIdx.y;
  int inPrev;
  int inNext;
  int inCurr;
  /*
  inPrev = A0(i, j, 0);
  inCurr = A0(i, j, 1);
  inNext = A0(i, j, 2);
  for (int k = 1; k < nz - 1; ++k) {
    if (j >= 1 && j < ny - 1 && i >= 1 && i < nx - 1) {
      Anext(i, j, k) = (-6 * inCurr + inPrev + inNext + A0(i, j - 1, k) + A0(i, j + 1, k) + A0(i - 1, j, k) + A0(i + 1, j, k));
    }
    
    inPrev = inCurr;
    inCurr = inNext;
    if (k+2 < nz)
      inNext = A0(i, j, k + 2);
  }
  */
  
  __shared__ int inCurr_s[TILE_SIZE][TILE_SIZE];
  // target is from 1 to n-2
  if (i >= 0 && i < nx && j >= 0 && j < ny) {
    inPrev                             = A0(i, j, 0);
    inCurr                             = A0(i, j, 1);
    inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
/*
        if(i==1&&j==0){
      printf("inCurr: %d\n", inCurr);
      printf("threadIdx.y: %d, threadIdx.x: %d\n", threadIdx.y, threadIdx.x);
      printf("inCurr_s[threadIdx.y][threadIdx.x]: %f\n", inCurr_s[threadIdx.y][threadIdx.x]);
    }
*/
  }
  for (int k = 1; k < nz - 1; ++k) {
    if (j >= 0 && j < ny && i >= 0 && i < nx) {
      inNext = A0(i, j, k + 1);
    }
    __syncthreads();
    if (j >= 1 && j < ny - 1 && i >= 1 && i < nx - 1) {
      //if(i==1&&j==1&&k==1){
        //printf("A0(i,j,k): %d, A0(i,j,k-1): %d, A0(i,j,k+1): %d, A0(i,j-1,k): %d, A0(i,j+1,k): %d, A0(i-1,j,k): %d, A0(i+1,j,k): %d\ninCurr: %d, inPrev: %d, inNext: %d, inCurr_s[threadIdx.y - 1][threadIdx.x]: %f, inCurr_s[threadIdx.y + 1][threadIdx.x]: %f, inCurr_s[threadIdx.y][threadIdx.x-1]: %f, inCurr_s[threadIdx.y][threadIdx.x+1]: %f\nthreadIdx.y: %d, threadIdx.x: %d\n", A0(i, j, k), A0(i, j, k - 1), A0(i, j, k + 1), A0(i, j - 1, k), A0(i, j + 1, k), A0(i - 1, j, k), A0(i + 1, j, k), inCurr, inPrev, inNext, inCurr_s[threadIdx.y - 1][threadIdx.x], inCurr_s[threadIdx.y + 1][threadIdx.x], inCurr_s[threadIdx.y][threadIdx.x - 1], inCurr_s[threadIdx.y][threadIdx.x + 1], threadIdx.y, threadIdx.x);
      //}
      Anext(i, j, k) = -6 * inCurr + inPrev + inNext +

                       (threadIdx.y >= 1 ? inCurr_s[threadIdx.y - 1][threadIdx.x] : A0(i, j - 1, k)) +
                       ((threadIdx.y + 1) < TILE_SIZE ? inCurr_s[threadIdx.y + 1][threadIdx.x] : A0(i, j + 1, k)) +
                       (threadIdx.x >= 1 ? inCurr_s[threadIdx.y][threadIdx.x - 1] : A0(i - 1, j, k)) +
                       ((threadIdx.x + 1) < TILE_SIZE ? inCurr_s[threadIdx.y][threadIdx.x + 1] : A0(i + 1, j, k));

    }
    __syncthreads();
    inPrev                             = inCurr;
    inCurr                             = inNext;
    inCurr_s[threadIdx.y][threadIdx.x] = inNext;
  }
  
#undef A0
#undef Anext
}

void launchStencil(int *A0, int *Anext, int nx, int ny, int nz) {

  // INSERT CODE HERE
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
  dim3 dimGrid((nx + TILE_SIZE - 1) / TILE_SIZE, (ny + TILE_SIZE - 1) / TILE_SIZE, 1);
  kernel<<<dimGrid, dimBlock>>>(A0, Anext, nx, ny, nz);
}
static uint_fast32_t rng_uint32(uint_fast32_t *rng_state) {
  uint_fast32_t local = *rng_state;
  local ^= local << 13; // a
  local ^= local >> 17; // b
  local ^= local << 5;  // c
  *rng_state = local;
  return local;
}

static uint_fast32_t *rng_new_state(uint_fast32_t seed) {
  uint64_t *rng_state = new uint64_t;
  *rng_state          = seed;
  return rng_state;
}

static uint_fast32_t *rng_new_state() {
  return rng_new_state(88172645463325252LL);
}

static int rng_int(uint_fast32_t *state) {
  uint_fast32_t rnd = rng_uint32(state);
  return static_cast<int>(rnd);
}

static void generate_data(int *x, const int nx, const int ny, const int nz) {
  const auto rng_state = rng_new_state();

  for (size_t ii = 0; ii < nx * ny * nz; ++ii) {
    x[ii] = rng_int(rng_state);
  }

  delete rng_state;
}
#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << " in " << file << " at line " << line << std::endl;
    exit(-1);
  }
}

static bool verify(const int *Anext, const int *A0, const int nx, const int ny, const int nz) {

#define _Anext(xi, yi, zi) Anext[(zi) * (ny * nx) + (yi) *nx + (xi)]
#define _A0(xi, yi, zi) A0[(zi) * (ny * nx) + (yi) *nx + (xi)]

  for (size_t z = 1; z < nz - 1; ++z) {
    for (size_t y = 1; y < ny - 1; ++y) {
      for (size_t x = 1; x < nx - 1; ++x) {
        const int expected = _A0(x, y, z + 1) + _A0(x, y, z - 1) + _A0(x, y + 1, z) + _A0(x, y - 1, z) + _A0(x + 1, y, z) +
                             _A0(x - 1, y, z) - 6 * _A0(x, y, z);
        if (expected != _Anext(x, y, z)) {
          std::cerr << "the results did not match at [" << x << "," << y << "," << z << "]";
          std::cerr << "(expected == _Anext(x, y, z))" << expected << " != " << _Anext(x, y, z) << std::endl;
          exit(-1);
        }
      }
    }
  }

  return true;

#undef _Anext
#undef _A0
}

static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + std::to_string(ny) + "," + std::to_string(nz) + "]";

  // generate input data
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **) &deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **) &deviceAnext, nx * ny * nz * sizeof(int)));

  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());

  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());

  // verify with provided implementation
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}
int main() {
  eval(32, 32, 32);
}