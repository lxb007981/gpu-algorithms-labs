/*! \brief File containing evaluation and grading code

This file contains evaluation and grading code.
This code is in a separate file so that a known-good version can be used for
automatic grading and students should not need to modify it.
*/

#include "assert.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <random>
#include <string>

#include "template.hu"

namespace gpu_algorithms_labs_evaluation {

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

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << " in " << file << " at line " << line << std::endl;
    exit(-1);
  }
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
    }
  }
}

static int eval(const size_t matArow, const size_t matAcol, const size_t matBcol) {

  const size_t matBrow = matAcol;

  // Generate model
  const auto conf_info = std::string("sgemm[<") + std::to_string(matArow) + "," + std::to_string(matAcol) + ">x<" +
                         std::to_string(matBrow) + "," + std::to_string(matBcol) + ">]";

  const size_t aSz = matArow * matAcol;
  const size_t bSz = matBrow * matBcol;
  const size_t cSz = matArow * matBcol;

  // generate input data
  std::vector<float> hostA(aSz);
  std::vector<float> hostB(bSz);
  std::vector<float> hostC(cSz);
  generate_data(hostA.data(), hostA.size());
  generate_data(hostB.data(), hostB.size());

  float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **) &deviceA, aSz * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc((void **) &deviceB, bSz * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc((void **) &deviceC, cSz * sizeof(float)));

  CUDA_RUNTIME(cudaMemcpy(deviceA, hostA.data(), aSz * sizeof(float), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaMemcpy(deviceB, hostB.data(), bSz * sizeof(float), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());

  //////////////////////////////////////////
  // GPU Stencil Computation
  //////////////////////////////////////////
  basicSgemm('N', 'T', matArow, matBcol, matBrow, 1.0f, deviceA, matArow, deviceB, matBrow, 0.0f, deviceC, matBrow);
  CUDA_RUNTIME(cudaDeviceSynchronize());

  CUDA_RUNTIME(cudaMemcpy(hostC.data(), deviceC, cSz * sizeof(float), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());

  // verify with provided implementation
  verify(hostA.data(), hostB.data(), hostC.data(), matArow, matAcol, matBcol);

  CUDA_RUNTIME(cudaFree(deviceA));
  CUDA_RUNTIME(cudaFree(deviceB));
  CUDA_RUNTIME(cudaFree(deviceC));

  return 0;
}

} // namespace gpu_algorithms_labs_evaluation

int main() {
  gpu_algorithms_labs_evaluation::eval(31,31,31);
}
