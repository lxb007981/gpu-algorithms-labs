#include "helper.hpp"
__constant__ float filter[800];
#define TILE_DIM 4
#define CHANNELS 1
// X[b, c, h+p, w+q] = X[((b * xdims.depth + c) * xdims.height + (h + p)) * xdims.width + (w + q)]
// W[m, c, p, q] = W[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q]
// Y[b, m, h, w] = Y[((b * ydims.depth + m) * ydims.height + h) * ydims.width + w]

// Sequential code for the forward path of the convolution layer
// You should not modify this code
static void conv_forward_valid(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {
  std::fill(Y, Y + ydims.flattened_length(), 0);

  for (auto i : range(0, ydims.num)) {
    for (auto m : range(0, ydims.depth)) {    // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width)) {
          const auto yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth)) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {  // filter height
              for (auto q : range(0, wdims.width)) { // filter width
                const auto xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const auto woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Baseline GPU kernel code for forward convolution.
// One thread per output index
// You should not modify this kernel as it is used for correctness comparison.
// Instead, define a new one below
__global__ void conv_forward_baseline_kernel(const float *X, const shape xdims, const shape wdims, float *Y, const shape ydims) {

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < ydims.num * ydims.depth * ydims.height * ydims.width; i += blockDim.x * gridDim.x) {
    Y[i] = 0.f;
  }

  for (size_t i = gx; i < ydims.num; i += gridDim.x * blockDim.x) {
    for (auto m : range(0, ydims.depth)) {    // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width)) {
          const size_t yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          float sum            = 0;
          for (auto c : range(0, xdims.depth)) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {  // filter height
              for (auto q : range(0, wdims.width)) { // filter width
                const size_t xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const size_t woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                sum += X[xoffset] * filter[woffset];
              }
            }
          }
          Y[yoffset] = sum;
         }
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_baseline(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {

  dim3 dimGrid(1);
  dim3 dimBlock(32);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());
}

// Implement your optimized kernel here.
// Make any modifications you wish.
// Don't forget to modify the host code below, if needed!
__global__ void conv_forward_opt_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  __shared__ float Ns[CHANNELS][TILE_DIM][TILE_DIM];
  for (size_t i = 0; i < ydims.num; i += 1) { // for all inputs images
    for (auto m : range(0, ydims.depth))      // for each output feature map
    {
      // load the input tile into shared memory
      if (row < ydims.height && col < ydims.width) {
        for (auto c : range(0, xdims.depth)) { // for each input feature map
          int xoffset                     = ((((i * xdims.depth) + c) * xdims.height) + row) * xdims.width + col;
          Ns[c][threadIdx.y][threadIdx.x] = X[xoffset];
        }
      } else {
        for (auto c : range(0, xdims.depth)) { // for each input feature map
          Ns[c][threadIdx.y][threadIdx.x] = 0;
        }
      }
      __syncthreads();
      // compute the convolution
      float sum = 0;
      if (row < ydims.height && col < ydims.width) {
        for (auto c : range(0, xdims.depth)) {     // sum over all input feature maps
          for (auto p : range(0, wdims.height)) {  // filter height
            for (auto q : range(0, wdims.width)) { // filter width
                                                   // if the input should be loaded from shared memory or from global memory
              int x = (signed)threadIdx.x - (signed)wdims.width / 2 + q;
              int y = (signed)threadIdx.y - (signed)wdims.height / 2 + p;
              if (x >= 0 && x < TILE_DIM &&
                  y >= 0 && y < TILE_DIM) {
                sum += Ns[c][threadIdx.y + p][threadIdx.x + q] * filter[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q];

              } else {
                int xoffset = ((((i * xdims.depth) + c) * xdims.height) + (row + p)) * xdims.width + (col + q);
                sum += X[xoffset] * filter[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q];
              }
            }
          }
        }
        int yoffset = ((i * ydims.depth + m) * ydims.height + row) * ydims.width + col;
        Y[yoffset]  = sum;
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {
  // initialize filter in constant memory
  cudaMemcpyToSymbol(filter, W, wdims.flattened_length() * sizeof(float));

  //@@ YOUR CODE HERE
  dim3 dimGrid((ydims.width + TILE_DIM - 1) / TILE_DIM, (ydims.height + TILE_DIM - 1) / TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());
}

static int eval(const shape wDims, const shape xDims) {

  // Generate model
  const auto conf_info = std::string("conv[wDims:") + std::to_string(wDims.num) + "," + std::to_string(wDims.depth) + "," +
                         std::to_string(wDims.height) + "," + std::to_string(wDims.width) + " xDims:" + std::to_string(xDims.num) + "," +
                         std::to_string(xDims.depth) + "," + std::to_string(xDims.height) + "," + std::to_string(xDims.width) + "]";
  INFO("Running " << conf_info);

  // Generate convolution weights
  float *hostW = allocate<float>(wDims);
  generate_convfilters(hostW, wDims);

  // generate input feature map
  float *hostX = allocate<float>(xDims);
  generate_data(hostX, xDims);

  // generate output feature map for verification
  const shape ydims = {xDims.num, wDims.num, (xDims.height - wDims.height + 1), (xDims.width - wDims.width + 1)};
  INFO("Allocating output tensor [" << ydims.num << "," << ydims.depth << "," << ydims.height << "," << ydims.width << "]");
  float *hostY    = allocate<float>(ydims);
  float *expected = allocate<float>(ydims);
  generate_data(hostY, ydims);

  const size_t wByteCount = wDims.flattened_length() * sizeof(float);
  const size_t xByteCount = xDims.flattened_length() * sizeof(float);
  const size_t yByteCount = ydims.flattened_length() * sizeof(float);

  float *deviceW = nullptr, *deviceX = nullptr, *deviceY = nullptr;
  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **) &deviceW, wByteCount));
  THROW_IF_ERROR(cudaMalloc((void **) &deviceX, xByteCount));
  THROW_IF_ERROR(cudaMalloc((void **) &deviceY, yByteCount));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceW, hostW, wByteCount, cudaMemcpyDefault));
  THROW_IF_ERROR(cudaMemcpy(deviceX, hostX, xByteCount, cudaMemcpyDefault));
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  convlayer_gpu_opt(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  THROW_IF_ERROR(cudaMemcpy(hostY, deviceY, yByteCount, cudaMemcpyDefault));
  timer_stop();

  // verify with provided implementation
  convlayer_gpu_baseline(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  THROW_IF_ERROR(cudaMemcpy(expected, deviceY, yByteCount, cudaMemcpyDefault));
  verify(expected, hostY, ydims);

  THROW_IF_ERROR(cudaFree(deviceW));
  THROW_IF_ERROR(cudaFree(deviceX));
  THROW_IF_ERROR(cudaFree(deviceY));
  free(hostW);
  free(hostX);
  free(hostY);
  free(expected);

  return 0;
}

TEST_CASE("Convlayer", "[convlayer]") {
  SECTION("[wDims:0,0,0,0 xDims:100,1,32,32]") {
    eval({0, 0, 0, 0}, {100, 1, 32, 32});
  }
  SECTION("[wDims:1,1,1,1 xDims:100,1,32,32]") {
    eval({1, 1, 1, 1}, {100, 1, 32, 32});
  }
  SECTION("[wDims:32,1,5,5 xDims:1000,1,28,28]") {
    eval({32, 1, 5, 5}, {1000, 1, 28, 28});
  }
  SECTION("[wDims:16,1,3,3 xDims:100,1,32,32]") {
    eval({16, 1, 3, 3}, {100, 1, 32, 32});
  }
}
