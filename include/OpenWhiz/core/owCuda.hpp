/*
 * owCuda.hpp
 *
 *  Created on: Apr 21, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once

#ifdef OW_USE_GPU
#include <cuda_runtime.h>
#include <iostream>

namespace ow {
namespace cuda {

/**
 * @brief Wrapper for Linear Layer Forward Pass on GPU via CUDA
 * Implementation in owCuda.cu
 */
void linearForward(const float* input, const float* weights, const float* bias, float* output,
                  int batchSize, int inputSize, int outputSize);

} // namespace cuda
} // namespace ow

#endif
