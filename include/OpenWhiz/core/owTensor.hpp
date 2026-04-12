/*
 * owTensor.hpp
 *
 *  Created on: Nov 24, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once

#include <vector>
#include <array>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <string>
#include <type_traits>
#include <sstream>
#include <algorithm>
#include "owSimd.hpp"
#include <initializer_list>
#include <chrono>

/**
 * @file owTensor.hpp
 * @brief High-performance N-dimensional tensor implementation for OpenWhiz.
 * 
 * This file contains the core tensor structures used throughout the OpenWhiz library.
 * It supports both numerical (float, double, int) and string data types within a 
 * unified template-based architecture.
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#define OW_ALIGNED_MALLOC(size, alignment) _aligned_malloc(size, alignment)
#define OW_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
/**
 * @brief Helper for POSIX-compliant aligned memory allocation.
 * @param size Total bytes to allocate.
 * @param alignment Alignment boundary (must be power of 2).
 * @return Pointer to allocated memory.
 */
static inline void* ow_posix_aligned_alloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
}
#define OW_ALIGNED_MALLOC(size, alignment) ow_posix_aligned_alloc(size, alignment)
#define OW_ALIGNED_FREE(ptr) free(ptr)
#endif

namespace ow {

/**
 * @class owTensor
 * @brief Template class for multi-dimensional arrays (tensors) with optional SIMD acceleration.
 * 
 * owTensor is the fundamental data structure in OpenWhiz. It provides a flexible and 
 * efficient way to handle N-dimensional data, used for neural network weights, 
 * activations, datasets, and even metadata (as string tensors).
 * 
 * @tparam T The data type stored in the tensor. Supported types include:
 *           - Numerical: float, double, int (accelerated with SIMD).
 *           - Textual: std::string (integrated from the former owStringTensor).
 * @tparam Rank The dimensionality of the tensor (e.g., 1 for Vector, 2 for Matrix).
 * 
 * @details
 * **Key Features:**
 * - **Aligned Memory:** Automatically uses 64-byte alignment to support modern SIMD (AVX-512, AVX2).
 * - **SIMD Acceleration:** specialized arithmetic operators for 'float' using CPU-specific intrinsics.
 * - **Multi-threading:** Integrated OpenMP support for large-scale operations.
 * - **Generic Design:** Seamlessly handles complex types like std::string through template specialization.
 * - **Tiled Matrix Multiplication:** Optimized 'dot' product to maximize CPU cache efficiency.
 */
template <typename T, size_t Rank>
class owTensor {
public:
    /** @brief Type alias for the tensor's shape array. */
    using owTensorShape = std::array<size_t, Rank>;

    /**
     * @brief Default constructor. Creates an empty tensor with no allocated memory.
     */
    owTensor() : m_size(0), m_data_ptr(nullptr), m_owns_data(false) { m_shape.fill(0); }
    
    /**
     * @brief Variadic constructor to define shape during instantiation.
     * @param args Dimensions for each rank (e.g., owTensor<float, 2>(rows, cols)).
     */
    template<typename... Args, typename std::enable_if<sizeof...(Args) == Rank, int>::type = 0>
    owTensor(Args... args) : m_owns_data(true) {
        size_t dims[] = { static_cast<size_t>(args)... };
        for (size_t i = 0; i < Rank; ++i) m_shape[i] = dims[i];
        m_size = calculate_size(m_shape);
        m_data_ptr = allocate(m_size);
    }

    /**
     * @brief Specialized constructor for Rank 2 matrices with an initial value.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param init_val Initial value for all elements.
     */
    template<size_t R = Rank, typename std::enable_if<R == 2, int>::type = 0>
    owTensor(size_t rows, size_t cols, T init_val) : m_owns_data(true) {
        m_shape[0] = rows; m_shape[1] = cols;
        m_size = rows * cols;
        m_data_ptr = allocate(m_size);
        std::fill(m_data_ptr, m_data_ptr + m_size, init_val);
    }

    /**
     * @brief Constructs a tensor with a given shape.
     * @param shape std::array containing dimensions.
     */
    explicit owTensor(const owTensorShape& shape) : m_shape(shape), m_owns_data(true) {
        m_size = calculate_size(shape);
        m_data_ptr = allocate(m_size);
    }
    
    /**
     * @brief Constructs a tensor with a given shape and initial value.
     * @param shape std::array containing dimensions.
     * @param init_val Initial value for all elements.
     */
    owTensor(const owTensorShape& shape, T init_val) : m_shape(shape), m_owns_data(true) {
        m_size = calculate_size(shape);
        m_data_ptr = allocate(m_size);
        std::fill(m_data_ptr, m_data_ptr + m_size, init_val);
    }

    /**
     * @brief Constructs a tensor from an initializer list with a specific shape.
     * @param shape Target shape.
     * @param list Flattened data list.
     * @throws std::runtime_error If the list size does not match the product of shape dimensions.
     */
    owTensor(const owTensorShape& shape, std::initializer_list<T> list) : m_shape(shape), m_owns_data(true) {
        m_size = calculate_size(shape);
        if (list.size() != m_size) {
            throw std::runtime_error("Initializer list size mismatch! Expected " + std::to_string(m_size) + " but got " + std::to_string(list.size()));
        }
        m_data_ptr = allocate(m_size);
        std::copy(list.begin(), list.end(), m_data_ptr);
    }

    /**
     * @brief Constructs a 1D tensor (or higher rank with last dim set) from an initializer list.
     * @param list Data list.
     */
    owTensor(std::initializer_list<T> list) : m_owns_data(true) {
        m_size = list.size();
        if (Rank == 1) {
            m_shape[0] = m_size;
        } else {
            m_shape.fill(1);
            m_shape[Rank - 1] = m_size;
        }
        m_data_ptr = allocate(m_size);
        std::copy(list.begin(), list.end(), m_data_ptr);
    }
    
    /**
     * @brief Copy constructor. Performs a deep copy if the source owns its data.
     * @param other Source tensor to copy from.
     */
    owTensor(const owTensor& other) : m_shape(other.m_shape), m_size(other.m_size), m_owns_data(other.m_owns_data) {
        if (other.m_owns_data) {
            m_data_ptr = allocate(m_size);
            std::copy(other.m_data_ptr, other.m_data_ptr + m_size, m_data_ptr);
        } else {
            m_data_ptr = other.m_data_ptr;
        }
    }

    /**
     * @brief Move constructor. Efficiently transfers ownership of the underlying buffer.
     * @param other Source tensor to move from.
     */
    owTensor(owTensor&& other) noexcept : m_shape(other.m_shape), m_size(other.m_size), m_data_ptr(other.m_data_ptr), m_owns_data(other.m_owns_data) {
        other.m_data_ptr = nullptr;
        other.m_owns_data = false;
        other.m_size = 0;
    }

    /**
     * @brief Copy assignment operator. Frees existing data and performs a deep copy.
     * @param other Source tensor.
     * @return Reference to this tensor.
     */
    owTensor& operator=(const owTensor& other) {
        if (this != &other) {
            if (m_owns_data && m_data_ptr) OW_ALIGNED_FREE(m_data_ptr);
            m_shape = other.m_shape;
            m_size = other.m_size;
            m_owns_data = other.m_owns_data;
            if (m_owns_data) {
                m_data_ptr = allocate(m_size);
                std::copy(other.m_data_ptr, other.m_data_ptr + m_size, m_data_ptr);
            } else {
                m_data_ptr = other.m_data_ptr;
            }
        }
        return *this;
    }

    /**
     * @brief Move assignment operator.
     * @param other Source tensor.
     * @return Reference to this tensor.
     */
    owTensor& operator=(owTensor&& other) noexcept {
        if (this != &other) {
            if (m_owns_data && m_data_ptr) OW_ALIGNED_FREE(m_data_ptr);
            m_shape = other.m_shape;
            m_size = other.m_size;
            m_data_ptr = other.m_data_ptr;
            m_owns_data = other.m_owns_data;
            other.m_data_ptr = nullptr;
            other.m_owns_data = false;
            other.m_size = 0;
        }
        return *this;
    }

    /**
     * @brief Destructor. Ensures proper destruction of non-trivial types (like std::string) 
     *        before freeing aligned memory.
     */
    virtual ~owTensor() {
        if (m_owns_data && m_data_ptr) {
            if (!std::is_trivially_destructible<T>::value) {
                for (size_t i = 0; i < m_size; ++i) m_data_ptr[i].~T();
            }
            OW_ALIGNED_FREE(m_data_ptr);
        }
    }

    /**
     * @brief Static helper to create a tensor filled with zeros.
     * @tparam U For SFINAE arithmetic check.
     * @param shape Desired shape.
     * @return New owTensor.
     */
    template<typename U = T>
    static typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    Zeros(const owTensorShape& shape) { return owTensor(shape, static_cast<T>(0)); }

    /**
     * @brief Static helper to create a tensor filled with ones.
     * @tparam U For SFINAE arithmetic check.
     * @param shape Desired shape.
     * @return New owTensor.
     */
    template<typename U = T>
    static typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    Ones(const owTensorShape& shape) { return owTensor(shape, static_cast<T>(1)); }

    /**
     * @brief Static helper to create a tensor with random values.
     * @tparam U For SFINAE arithmetic check.
     * @param shape Desired shape.
     * @param min Minimum range value.
     * @param max Maximum range value.
     * @return New owTensor with random data.
     */
    template<typename U = T>
    static typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    Random(const owTensorShape& shape, T min = -1.0, T max = 1.0) {
        owTensor tensor(shape); tensor.setRandom(min, max); return tensor;
    }

    /**
     * @brief Fills the entire tensor with zeros.
     * @tparam U For SFINAE arithmetic check.
     */
    template<typename U = T, typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
    void setZero() { if (m_data_ptr) std::fill(m_data_ptr, m_data_ptr + m_size, static_cast<T>(0)); }

    /**
     * @brief Fills the entire tensor with a constant value.
     * @param val Value to apply to all elements.
     */
    void setConstant(T val) { if (m_data_ptr) std::fill(m_data_ptr, m_data_ptr + m_size, val); }
    
    /**
     * @brief Sets values from an initializer list.
     * @param list Data list.
     * @throws std::runtime_error If list size does not match tensor size.
     */
    void setValues(std::initializer_list<T> list) {
        if (list.size() != m_size) {
            throw std::runtime_error("Initializer list size mismatch! Expected " + std::to_string(m_size) + " but got " + std::to_string(list.size()));
        }
        std::copy(list.begin(), list.end(), m_data_ptr);
    }

    /**
     * @brief Sets values from a nested initializer list (for Rank 2 tensors).
     * @param list Nested list {{r1c1, r1c2}, {r2c1, r2c2}}.
     */
    template<size_t R = Rank, typename std::enable_if<R == 2, int>::type = 0>
    void setValues(std::initializer_list<std::initializer_list<T>> list) {
        if (list.size() != m_shape[0]) throw std::runtime_error("Row count mismatch!");
        size_t row = 0;
        for (const auto& row_list : list) {
            if (row_list.size() != m_shape[1]) throw std::runtime_error("Column count mismatch!");
            std::copy(row_list.begin(), row_list.end(), m_data_ptr + row * m_shape[1]);
            row++;
        }
    }

    /**
     * @brief Populates the tensor with random values using a uniform distribution.
     * @tparam U For SFINAE arithmetic check.
     * @param min Minimum range.
     * @param max Maximum range.
     */
    template<typename U = T, typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
    void setRandom(T min = -1.0, T max = 1.0) {
        if (!m_data_ptr) return;
        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(static_cast<double>(min), static_cast<double>(max));
        for (size_t i = 0; i < m_size; ++i) m_data_ptr[i] = static_cast<T>(dis(gen));
    }

    /**
     * @brief Multi-dimensional accessor using a shape array.
     * @param indices Array containing the index for each dimension.
     * @return Reference to the element.
     */
    T& operator()(const owTensorShape& indices) { return m_data_ptr[get_index(indices)]; }
    /** @brief Multi-dimensional const accessor. */
    const T& operator()(const owTensorShape& indices) const { return m_data_ptr[get_index(indices)]; }

    /**
     * @brief 1D accessor for Rank 1 tensors.
     * @param i Index.
     */
    template<size_t R = Rank> typename std::enable_if<R == 1, T&>::type operator()(size_t i) { return m_data_ptr[i]; }
    /** @brief 1D const accessor for Rank 1 tensors. */
    template<size_t R = Rank> typename std::enable_if<R == 1, const T&>::type operator()(size_t i) const { return m_data_ptr[i]; }

    /**
     * @brief 2D accessor for Rank 2 matrices.
     * @param i Row index.
     * @param j Column index.
     */
    template<size_t R = Rank> typename std::enable_if<R == 2, T&>::type operator()(size_t i, size_t j) { return m_data_ptr[i * m_shape[1] + j]; }
    /** @brief 2D const accessor for Rank 2 matrices. */
    template<size_t R = Rank> typename std::enable_if<R == 2, const T&>::type operator()(size_t i, size_t j) const { return m_data_ptr[i * m_shape[1] + j]; }

    /**
     * @brief Element-wise addition with SIMD acceleration for float.
     * @param other Tensor to add.
     * @return Resulting tensor.
     */
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator+(const owTensor& other) const {
        check_shape(other); owTensor res(m_shape);
        T* r = res.m_data_ptr; const T* a = m_data_ptr; const T* b = other.m_data_ptr;
        
        if (std::is_same<T, float>::value) {
            #ifdef __AVX512F__
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~15); i += 16) {
                _mm512_storeu_ps(reinterpret_cast<float*>(r + i), _mm512_add_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(a + i)), _mm512_loadu_ps(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~15); i < m_size; ++i) r[i] = a[i] + b[i];
            #elif defined(__AVX2__)
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~7); i += 8) {
                _mm256_storeu_ps(reinterpret_cast<float*>(r + i), _mm256_add_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(a + i)), _mm256_loadu_ps(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~7); i < m_size; ++i) r[i] = a[i] + b[i];
            #elif defined(OW_ARM_NEON)
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~3); i += 4) {
                vst1q_f32(reinterpret_cast<float*>(r + i), vaddq_f32(vld1q_f32(reinterpret_cast<const float*>(a + i)), vld1q_f32(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~3); i < m_size; ++i) r[i] = a[i] + b[i];
            #else
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] + b[i];
            #endif
        } else {
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] + b[i];
        }
        return res;
    }

    /**
     * @brief Element-wise subtraction with SIMD acceleration for float.
     * @param other Tensor to subtract.
     * @return Resulting tensor.
     */
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator-(const owTensor& other) const {
        check_shape(other); owTensor res(m_shape);
        T* r = res.m_data_ptr; const T* a = m_data_ptr; const T* b = other.m_data_ptr;
        if (std::is_same<T, float>::value) {
            #ifdef __AVX512F__
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~15); i += 16) {
                _mm512_storeu_ps(reinterpret_cast<float*>(r + i), _mm512_sub_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(a + i)), _mm512_loadu_ps(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~15); i < m_size; ++i) r[i] = a[i] - b[i];
            #elif defined(__AVX2__)
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~7); i += 8) {
                _mm256_storeu_ps(reinterpret_cast<float*>(r + i), _mm256_sub_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(a + i)), _mm256_loadu_ps(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~7); i < m_size; ++i) r[i] = a[i] - b[i];
            #elif defined(OW_ARM_NEON)
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~3); i += 4) {
                vst1q_f32(reinterpret_cast<float*>(r + i), vsubq_f32(vld1q_f32(reinterpret_cast<const float*>(a + i)), vld1q_f32(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~3); i < m_size; ++i) r[i] = a[i] - b[i];
            #else
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] - b[i];
            #endif
        } else {
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] - b[i];
        }
        return res;
    }

    /**
     * @brief Element-wise multiplication (Hadamard product) with SIMD acceleration for float.
     * @param other Tensor to multiply with.
     * @return Resulting tensor.
     */
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator*(const owTensor& other) const {
        check_shape(other); owTensor res(m_shape);
        T* r = res.m_data_ptr; const T* a = m_data_ptr; const T* b = other.m_data_ptr;
        if (std::is_same<T, float>::value) {
            #ifdef __AVX512F__
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~15); i += 16) {
                _mm512_storeu_ps(reinterpret_cast<float*>(r + i), _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(a + i)), _mm512_loadu_ps(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~15); i < m_size; ++i) r[i] = a[i] * b[i];
            #elif defined(__AVX2__)
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~7); i += 8) {
                _mm256_storeu_ps(reinterpret_cast<float*>(r + i), _mm256_mul_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(a + i)), _mm256_loadu_ps(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~7); i < m_size; ++i) r[i] = a[i] * b[i];
            #elif defined(OW_ARM_NEON)
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~3); i += 4) {
                vst1q_f32(reinterpret_cast<float*>(r + i), vmulq_f32(vld1q_f32(reinterpret_cast<const float*>(a + i)), vld1q_f32(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~3); i < m_size; ++i) r[i] = a[i] * b[i];
            #else
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] * b[i];
            #endif
        } else {
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] * b[i];
        }
        return res;
    }

    /**
     * @brief Element-wise division with SIMD acceleration for float.
     * @param other Tensor to divide by.
     * @return Resulting tensor.
     */
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator/(const owTensor& other) const {
        check_shape(other); owTensor res(m_shape);
        T* r = res.m_data_ptr; const T* a = m_data_ptr; const T* b = other.m_data_ptr;
        if (std::is_same<T, float>::value) {
            #ifdef __AVX512F__
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~15); i += 16) {
                _mm512_storeu_ps(reinterpret_cast<float*>(r + i), _mm512_div_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(a + i)), _mm512_loadu_ps(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~15); i < m_size; ++i) r[i] = a[i] / b[i];
            #elif defined(__AVX2__)
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~7); i += 8) {
                _mm256_storeu_ps(reinterpret_cast<float*>(r + i), _mm256_div_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(a + i)), _mm256_loadu_ps(reinterpret_cast<const float*>(b + i))));
            }
            for (size_t i = (m_size & ~7); i < m_size; ++i) r[i] = a[i] / b[i];
            #elif defined(OW_ARM_NEON)
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~3); i += 4) {
                #if defined(__aarch64__)
                vst1q_f32(reinterpret_cast<float*>(r + i), vdivq_f32(vld1q_f32(reinterpret_cast<const float*>(a + i)), vld1q_f32(reinterpret_cast<const float*>(b + i))));
                #else
                for(int j=0; j<4; ++j) r[i+j] = a[i+j] / b[i+j];
                #endif
            }
            for (size_t i = (m_size & ~3); i < m_size; ++i) r[i] = a[i] / b[i];
            #else
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] / b[i];
            #endif
        } else {
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] / b[i];
        }
        return res;
    }

    /**
     * @brief Multiplies every element by a scalar.
     * @param scalar The scalar value.
     * @return Resulting tensor.
     */
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator*(T scalar) const {
        owTensor res(m_shape);
        T* r = res.m_data_ptr;
        const T* a = m_data_ptr;
        if (std::is_same<T, float>::value) {
            #ifdef __AVX512F__
            __m512 v_scalar = _mm512_set1_ps(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~15); i += 16) {
                _mm512_storeu_ps(reinterpret_cast<float*>(r + i), _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(a + i)), v_scalar));
            }
            for (size_t i = (m_size & ~15); i < m_size; ++i) r[i] = a[i] * scalar;
            #elif defined(__AVX2__)
            __m256 v_scalar = _mm256_set1_ps(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~7); i += 8) {
                _mm256_storeu_ps(reinterpret_cast<float*>(r + i), _mm256_mul_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(a + i)), v_scalar));
            }
            for (size_t i = (m_size & ~7); i < m_size; ++i) r[i] = a[i] * scalar;
            #elif defined(OW_ARM_NEON)
            float32x4_t v_scalar = vdupq_n_f32(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~3); i += 4) {
                vst1q_f32(reinterpret_cast<float*>(r + i), vmulq_f32(vld1q_f32(reinterpret_cast<const float*>(a + i)), v_scalar));
            }
            for (size_t i = (m_size & ~3); i < m_size; ++i) r[i] = a[i] * scalar;
            #else
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] * scalar;
            #endif
        } else {
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] * scalar;
        }
        return res;
    }

    /**
     * @brief Divides every element by a scalar.
     * @param scalar The scalar value.
     * @return Resulting tensor.
     * @throws std::runtime_error If scalar is zero.
     */
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator/(T scalar) const {
        if (scalar == static_cast<T>(0)) throw std::runtime_error("Division by zero!");
        owTensor res(m_shape);
        T* r = res.m_data_ptr;
        const T* a = m_data_ptr;
        if (std::is_same<T, float>::value) {
            #ifdef __AVX512F__
            __m512 v_scalar = _mm512_set1_ps(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~15); i += 16) {
                _mm512_storeu_ps(reinterpret_cast<float*>(r + i), _mm512_div_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(a + i)), v_scalar));
            }
            for (size_t i = (m_size & ~15); i < m_size; ++i) r[i] = a[i] / scalar;
            #elif defined(__AVX2__)
            __m256 v_scalar = _mm256_set1_ps(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~7); i += 8) {
                _mm256_storeu_ps(reinterpret_cast<float*>(r + i), _mm256_div_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(a + i)), v_scalar));
            }
            for (size_t i = (m_size & ~7); i < m_size; ++i) r[i] = a[i] / scalar;
            #elif defined(OW_ARM_NEON)
            float32x4_t v_scalar = vdupq_n_f32(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~3); i += 4) {
                #if defined(__aarch64__)
                vst1q_f32(reinterpret_cast<float*>(r + i), vdivq_f32(vld1q_f32(reinterpret_cast<const float*>(a + i)), v_scalar));
                #else
                for(int j=0; j<4; ++j) r[i+j] = a[i+j] / scalar;
                #endif
            }
            for (size_t i = (m_size & ~3); i < m_size; ++i) r[i] = a[i] / scalar;
            #else
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] / scalar;
            #endif
        } else {
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] / scalar;
        }
        return res;
    }

    /**
     * @brief Adds a scalar to every element.
     * @param scalar The scalar value.
     * @return Resulting tensor.
     */
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator+(T scalar) const {
        owTensor res(m_shape);
        T* r = res.m_data_ptr;
        const T* a = m_data_ptr;
        if (std::is_same<T, float>::value) {
            #ifdef __AVX512F__
            __m512 v_scalar = _mm512_set1_ps(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~15); i += 16) {
                _mm512_storeu_ps(reinterpret_cast<float*>(r + i), _mm512_add_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(a + i)), v_scalar));
            }
            for (size_t i = (m_size & ~15); i < m_size; ++i) r[i] = a[i] + scalar;
            #elif defined(__AVX2__)
            __m256 v_scalar = _mm256_set1_ps(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~7); i += 8) {
                _mm256_storeu_ps(reinterpret_cast<float*>(r + i), _mm256_add_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(a + i)), v_scalar));
            }
            for (size_t i = (m_size & ~7); i < m_size; ++i) r[i] = a[i] + scalar;
            #elif defined(OW_ARM_NEON)
            float32x4_t v_scalar = vdupq_n_f32(static_cast<float>(scalar));
            #pragma omp parallel for if(m_size > 1000)
            for (long long i = 0; i < (long long)(m_size & ~3); i += 4) {
                vst1q_f32(reinterpret_cast<float*>(r + i), vaddq_f32(vld1q_f32(reinterpret_cast<const float*>(a + i)), v_scalar));
            }
            for (size_t i = (m_size & ~3); i < m_size; ++i) r[i] = a[i] + scalar;
            #else
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] + scalar;
            #endif
        } else {
            #pragma omp parallel for simd if(m_size > 1000)
            for (long long i = 0; i < (long long)m_size; ++i) r[i] = a[i] + scalar;
        }
        return res;
    }

    /**
     * @brief Subtracts a scalar from every element.
     * @param scalar The scalar value.
     * @return Resulting tensor.
     */
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator-(T scalar) const {
        return (*this) + (-scalar);
    }

    /** @brief Friend operator for scalar-tensor multiplication. */
    template<typename U = T>
    friend typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator*(T scalar, const owTensor& tensor) { return tensor * scalar; }
    
    /** @brief Friend operator for scalar-tensor addition. */
    template<typename U = T>
    friend typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator+(T scalar, const owTensor& tensor) { return tensor + scalar; }
    
    /** @brief Friend operator for scalar-tensor subtraction. */
    template<typename U = T>
    friend typename std::enable_if<std::is_arithmetic<U>::value, owTensor>::type 
    operator-(T scalar, const owTensor& tensor) {
        owTensor res(tensor.m_shape);
        T* r = res.m_data_ptr;
        const T* a = tensor.m_data_ptr;
        #pragma omp parallel for simd if(tensor.m_size > 1000)
        for (long long i = 0; i < (long long)tensor.m_size; ++i) r[i] = scalar - a[i];
        return res;
    }

    /**
     * @brief Matrix multiplication (Dot product). Optimized only for Rank 2 tensors.
     * 
     * This implementation uses Tiling and SIMD (FMA) to achieve high performance on 
     * large matrices.
     * @param other The matrix to multiply with.
     * @return Result matrix of shape [rows_A, cols_B].
     * @throws std::runtime_error If columns of A do not match rows of B.
     */
    template<size_t R = Rank, typename U = T>
    typename std::enable_if<R == 2 && std::is_arithmetic<U>::value, owTensor<T, 2>>::type 
    dot(const owTensor<T, 2>& other) const {
        if (m_shape[1] != other.shape()[0]) throw std::runtime_error("Matrix dimension mismatch!");
        owTensorShape res_shape = {m_shape[0], other.shape()[1]};
        owTensor<T, 2> res(res_shape, static_cast<T>(0));
        const size_t M = m_shape[0];
        const size_t K = m_shape[1];
        const size_t N = other.shape()[1];
        const size_t tileSize = 32;
        #pragma omp parallel for collapse(2) if(M*N > 1024)
        for (size_t ii = 0; ii < M; ii += tileSize) {
            for (size_t jj = 0; jj < N; jj += tileSize) {
                for (size_t kk = 0; kk < K; kk += tileSize) {
                    for (size_t i = ii; i < std::min(ii + tileSize, M); ++i) {
                        for (size_t k = kk; k < std::min(kk + tileSize, K); ++k) {
                            T temp = (*this)(i, k);
                            T* res_row = &res(i, 0);
                            const T* other_row = &other(k, 0);
                            size_t j = jj;
                            size_t limit = std::min(jj + tileSize, N);
                            if (std::is_same<T, float>::value) {
                                #ifdef __AVX2__
                                __m256 v_temp = _mm256_set1_ps(static_cast<float>(temp));
                                for (; j + 7 < limit; j += 8) {
                                    _mm256_storeu_ps(reinterpret_cast<float*>(res_row + j), _mm256_fmadd_ps(v_temp, _mm256_loadu_ps(reinterpret_cast<const float*>(other_row + j)), _mm256_loadu_ps(reinterpret_cast<const float*>(res_row + j))));
                                }
                                #elif defined(OW_ARM_NEON)
                                float32x4_t v_temp = vdupq_n_f32(static_cast<float>(temp));
                                for (; j + 3 < limit; j += 4) {
                                    vst1q_f32(reinterpret_cast<float*>(res_row + j), vfmaq_f32(vld1q_f32(reinterpret_cast<const float*>(res_row + j)), v_temp, vld1q_f32(reinterpret_cast<const float*>(other_row + j))));
                                }
                                #endif
                            }
                            for (; j < limit; ++j) res_row[j] += temp * other_row[j];
                        }
                    }
                }
            }
        }
        return res;
    }

    /**
     * @brief Returns the shape of the tensor.
     * @return Const reference to the owTensorShape array.
     */
    const owTensorShape& shape() const { return m_shape; }

    /**
     * @brief Returns the total number of elements in the tensor.
     */
    size_t size() const { return m_size; }

    /**
     * @brief Provides raw pointer access to the internal aligned buffer.
     * @return Pointer to T.
     */
    T* data() { return m_data_ptr; }
    /** @brief Provides raw const pointer access to the internal aligned buffer. */
    const T* data() const { return m_data_ptr; }

    /**
     * @brief Performs matrix transpose. Specialized for Rank 2 matrices.
     * @return A new tensor with swapped dimensions and transposed data.
     */
    template<size_t R = Rank>
    typename std::enable_if<R == 2, owTensor<T, 2>>::type transpose() const {
        owTensorShape trans_shape = {m_shape[1], m_shape[0]};
        owTensor<T, 2> res(trans_shape);
        for (size_t i = 0; i < m_shape[0]; ++i) {
            for (size_t j = 0; j < m_shape[1]; ++j) res(j, i) = (*this)(i, j);
        }
        return res;
    }

    /**
     * @brief Prints the tensor metadata and formatted elements to standard output.
     *        Numerical values are printed with fixed precision; strings are quoted.
     */
    void print() const {
        std::cout << "owTensor (Rank " << Rank << ", owTensorShape: [";
        for (size_t i = 0; i < Rank; ++i) std::cout << m_shape[i] << (i < Rank - 1 ? ", " : "");
        std::cout << "])\n";
        print_dispatch(std::integral_constant<size_t, Rank>{});
    }

    /**
     * @brief Converts the entire tensor into a space-separated string.
     * @return String representation useful for serialization or debugging.
     */
    std::string toString() const {
        std::stringstream ss; ss << std::fixed << std::setprecision(8);
        for (size_t i = 0; i < m_size; ++i) ss << m_data_ptr[i] << (i == m_size - 1 ? "" : " ");
        return ss.str();
    }

    /**
     * @brief Populates the tensor from a space-separated string representation.
     * @param s The source string.
     */
    void fromString(const std::string& s) {
        std::stringstream ss(s); T val; size_t count = 0;
        while (ss >> val && count < m_size) m_data_ptr[count++] = val;
    }

    /**
     * @brief Utility function to calculate the product of dimensions.
     * @param shape The shape array.
     * @return Total number of elements.
     */
    static size_t calculate_size(const owTensorShape& shape) {
        size_t s = 1; for (auto dim : shape) s *= dim; return s;
    }

protected:
    owTensorShape m_shape;  ///< Array of dimensions.
    size_t m_size;         ///< Total element count.
    T* m_data_ptr;         ///< Pointer to aligned memory block.
    bool m_owns_data;      ///< Ownership flag for RAII.

    /**
     * @brief Internal memory allocator with 64-byte alignment.
     * @param size Number of elements (not bytes).
     * @return Pointer to allocated memory.
     */
    T* allocate(size_t size) {
        if (size == 0) return nullptr;
        T* ptr = static_cast<T*>(OW_ALIGNED_MALLOC(size * sizeof(T), 64));
        // Ensure constructor is called for non-trivial types (like std::string)
        if (!std::is_trivially_default_constructible<T>::value) {
            for (size_t i = 0; i < size; ++i) new (&ptr[i]) T();
        }
        return ptr;
    }

    /**
     * @brief Internal dispatch for printing Rank 2 matrices.
     */
    void print_dispatch(std::integral_constant<size_t, 2>) const {
        if (!m_data_ptr) return;
        for (size_t i = 0; i < m_shape[0]; ++i) {
            std::cout << "[ ";
            for (size_t j = 0; j < m_shape[1]; ++j) { print_element((*this)(i, j)); std::cout << " "; }
            std::cout << "]\n";
        }
    }

    /**
     * @brief Internal dispatch for printing generic tensors (Rank 1, 3+).
     */
    template<size_t R> void print_dispatch(std::integral_constant<size_t, R>) const {
        if (!m_data_ptr) return;
        std::cout << "[ ";
        for (size_t i = 0; i < m_size; ++i) { print_element(m_data_ptr[i]); std::cout << " "; }
        std::cout << "]\n";
    }

    /**
     * @brief Formats an individual element for printing based on its type.
     */
    template<typename U> void print_element(const U& val) const {
        if (std::is_floating_point<U>::value) std::cout << std::fixed << std::setprecision(4) << val;
        else if (std::is_same<U, std::string>::value) std::cout << "\"" << val << "\"";
        else std::cout << val;
    }

    /**
     * @brief Calculates the flat buffer index from multi-dimensional coordinates.
     */
    size_t get_index(const owTensorShape& indices) const {
        size_t idx = 0; size_t stride = 1;
        for (int i = Rank - 1; i >= 0; --i) { idx += indices[i] * stride; stride *= m_shape[i]; }
        return idx;
    }

    /**
     * @brief Validates that two tensors have identical shapes.
     * @throws std::runtime_error If shapes differ.
     */
    void check_shape(const owTensor& other) const { if (m_shape != other.m_shape) throw std::runtime_error("Tensor shape mismatch!"); }
};

/**
 * @class owTensorMap
 * @brief Zero-copy wrapper to treat existing memory as an owTensor.
 * 
 * owTensorMap allows performing high-performance tensor operations on external 
 * buffers (e.g., memory mapped files, UI framework data, or raw hardware pointers) 
 * without copying. It inherits all SIMD-accelerated operators from owTensor but 
 * does NOT manage the lifecycle of the data pointer.
 * 
 * @tparam T Data type.
 * @tparam Rank Tensor dimensionality.
 */
template <typename T, size_t Rank>
class owTensorMap : public owTensor<T, Rank> {
public:
    /**
     * @brief Wraps a raw data pointer into an owTensor interface.
     * @param data Pointer to the external buffer.
     * @param shape Desired shape for interpreting the buffer.
     */
    owTensorMap(T* data, const typename owTensor<T, Rank>::owTensorShape& shape) {
        this->m_shape = shape; 
        this->m_size = owTensor<T, Rank>::calculate_size(shape); 
        this->m_data_ptr = data;
        this->m_owns_data = false;
    }

    /**
     * @brief Default constructor for an uninitialized map.
     */
    owTensorMap() : owTensor<T, Rank>() { this->m_owns_data = false; }
};

} // namespace ow
