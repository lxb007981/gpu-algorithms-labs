#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

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
  
  __shared__ int inCurr_s[TILE_SIZE][TILE_SIZE];
  // target is from 1 to n-2
  if (i >= 0 && i < nx && j >= 0 && j < ny) {
    inPrev                             = A0(i, j, 0);
    inCurr                             = A0(i, j, 1);
    inCurr_s[threadIdx.y][threadIdx.x] = inCurr;

  }
  for (int k = 1; k < nz - 1; ++k) {
    if (j >= 0 && j < ny && i >= 0 && i < nx) {
      inNext = A0(i, j, k + 1);
    }
    __syncthreads();
    if (j >= 1 && j < ny - 1 && i >= 1 && i < nx - 1) {
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

static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + std::to_string(ny) + "," + std::to_string(nz) + "]";
  INFO("Running " << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **) &deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **) &deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}

TEST_CASE("Stencil", "[stencil]") {

  SECTION("[dims:32,32,32]") {
    eval(32, 32, 32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30, 30, 30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29, 29, 29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31, 31, 31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29, 29, 29);
  }
  SECTION("[dims:1,1,2]") {
    eval(1, 1, 2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512, 512, 64);
  }
}
