#include "helper.hpp"

__global__ void s2g_gpu_gather_kernel(uint32_t *in, uint32_t *out, int len) {
  //@@ INSERT KERNEL CODE HERE
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outIdx < len) {
    int out_reg = 0;
    for (int inIdx = 0; inIdx < len; ++inIdx) {
      int intermediate = outInvariant(in[inIdx]);
      out_reg += outDependent(intermediate, inIdx, outIdx);
    }
    out[outIdx] += out_reg;
  }
}


static void s2g_cpu_gather(uint32_t *in, uint32_t *out, int len) {
  for (int outIdx = 0; outIdx < len; ++outIdx) {
    int out_reg = 0;
    for (int inIdx = 0; inIdx < len; ++inIdx) {
      int intermediate = outInvariant(in[inIdx]);
      out_reg += outDependent(intermediate, inIdx, outIdx);
    }
    out[outIdx] += out_reg;
  }
}


static void s2g_gpu_gather(uint32_t *in, uint32_t *out, int len) {
  //@@ INSERT CODE HERE
  int blockSize = 256;
  int numBlocks = (len + blockSize - 1) / blockSize;
  s2g_gpu_gather_kernel<<<numBlocks, blockSize>>>(in, out, len);
}


static int eval(int inputLength) {
  uint32_t *deviceInput = nullptr;
  uint32_t *deviceOutput= nullptr;

  const std::string conf_info =
      std::string("gather[len:") + std::to_string(inputLength) + "]";
  INFO("Running "  << conf_info);

  auto hostInput = generate_input(inputLength);

  const size_t byteCount = inputLength * sizeof(uint32_t);

  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **)&deviceInput, byteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceOutput, byteCount));
  timer_stop();


  timer_start("Copying input memory to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceInput, hostInput.data(), byteCount,
                     cudaMemcpyHostToDevice));
  THROW_IF_ERROR(cudaMemset(deviceOutput, 0, byteCount));
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU Gather computation");
  s2g_gpu_gather(deviceInput, deviceOutput, inputLength);
  timer_stop();

  std::vector<uint32_t> hostOutput(inputLength);

  timer_start("Copying output memory to the CPU");
  THROW_IF_ERROR(cudaMemcpy(hostOutput.data(), deviceOutput, byteCount,
                     cudaMemcpyDeviceToHost));
  timer_stop();

  auto expected = compute_output(hostInput, inputLength);
  verify(expected, hostOutput);

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  return 0;
}



TEST_CASE("Gather", "[gather]") {
  SECTION("[inputSize:1024]") {
    eval(1024);
  }
  SECTION("[inputSize:2048]") {
    eval(2048);
  }
  SECTION("[inputSize:2047]") {
    eval(2047);
  }
  SECTION("[inputSize:2049]") {
    eval(2049);
  }
  SECTION("[inputSize:9101]") {
    eval(9101);
  }
  SECTION("[inputSize:9910]") {
    eval(9910);
  }
  SECTION("[inputSize:8192]") {
    eval(8192);
  }
  SECTION("[inputSize:8193]") {
    eval(8193);
  }
  SECTION("[inputSize:8191]") {
    eval(8191);
  }
  SECTION("[inputSize:16191]") {
    eval(16191);
  }
}
