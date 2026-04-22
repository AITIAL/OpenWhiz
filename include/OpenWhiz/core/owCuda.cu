#include "owCuda.hpp"
#include <device_launch_parameters.h>

#ifdef OW_USE_GPU

namespace ow {
namespace cuda {

/**
 * @brief Simple Matrix Multiplication Kernel: C = A * B + Bias
 */
__global__ void linearForwardKernel(const float* A, const float* B, const float* bias, float* C, 
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum + bias[col];
    }
}

/**
 * @brief Wrapper for Linear Layer Forward Pass on GPU
 */
void linearForward(const float* input, const float* weights, const float* bias, float* output,
                  int batchSize, int inputSize, int outputSize) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((outputSize + blockSize.x - 1) / blockSize.x, 
                  (batchSize + blockSize.y - 1) / blockSize.y);

    linearForwardKernel<<<gridSize, blockSize>>>(input, weights, bias, output, 
                                                 batchSize, inputSize, outputSize);
    
    // Check for kernel launch errors (Optional in Release, but good for stability)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Ensure GPU computation is finished before CPU reads the results
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace ow

#endif
