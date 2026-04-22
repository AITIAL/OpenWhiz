/*
 * owLinearLayer.hpp
 *
 *  Created on: Nov 24, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include "../optimizers/owOptimizer.hpp"
#include "../core/owCuda.hpp"
#include <random>
#include <chrono>

namespace ow {

/**
 * @class owLinearLayer
 * @brief Standard fully connected (dense) layer: y = xW + b.
 * 
 * The owLinearLayer is the foundational building block of most neural networks.
 * It performs a matrix multiplication of the input with a weight matrix and
 * adds a bias vector.
 * 
 * Implementation Details:
 * - Forward: output = activation(input * weights + bias).
 * - Initialization: Uses Xavier (Glorot) initialization by default to maintain
 *   signal variance across layers.
 * - Optimization: Internal loops are optimized using OpenMP (#pragma omp parallel for)
 *   for significant speedup on multi-core CPUs.
 * - Supports learnable weights and biases, regularization (L1/L2), and integrated
 *   activation functions.
 * 
 * Platform Notes:
 * - Computer: Best performance achieved with OpenMP enabled.
 * - Mobile: Efficient for small to medium layer sizes.
 * - Industrial: Robust implementation for general-purpose regression and classification.
 * 
 * Comparison:
 * - Unlike owAffineLayer (scalar), owLinearLayer is a vector-matrix operation.
 */
class owLinearLayer : public owLayer {
public:
    /**
     * @brief Constructor for owLinearLayer.
     * @param inputSize Number of input features (optional, can be auto-chained).
     * @param outputSize Number of output neurons.
     */
    owLinearLayer(size_t inputSize = 0, size_t outputSize = 1) 
        : m_inputSize(inputSize), m_outputSize(outputSize),
          m_params(m_inputSize * m_outputSize + m_outputSize),
          m_grads(m_inputSize * m_outputSize + m_outputSize),
          m_weights(m_params.data(), owTensorShape{m_inputSize, m_outputSize}),
          m_biases(m_params.data() + (m_inputSize * m_outputSize), owTensorShape{1, m_outputSize}),
          m_weightGradients(m_grads.data(), owTensorShape{m_inputSize, m_outputSize}),
          m_biasGradients(m_grads.data() + (m_inputSize * m_outputSize), owTensorShape{1, m_outputSize}) {
        m_layerName = "Linear Layer";
        if (m_inputSize > 0) initializeWeights();
    }

    /**
     * @brief Sets the input feature size and reinitializes parameters if needed.
     */
    void setInputSize(size_t size) override {
        if (m_inputSize == size) return;
        m_inputSize = size;
        m_params = owTensor<float, 1>(m_inputSize * m_outputSize + m_outputSize);
        m_grads = owTensor<float, 1>(m_inputSize * m_outputSize + m_outputSize);
        m_weights = owTensorMap<float, 2>(m_params.data(), owTensorShape{m_inputSize, m_outputSize});
        m_biases = owTensorMap<float, 2>(m_params.data() + (m_inputSize * m_outputSize), owTensorShape{1, m_outputSize});
        m_weightGradients = owTensorMap<float, 2>(m_grads.data(), owTensorShape{m_inputSize, m_outputSize});
        m_biasGradients = owTensorMap<float, 2>(m_grads.data() + (m_inputSize * m_outputSize), owTensorShape{1, m_outputSize});
        initializeWeights();
    }

    /**
     * @brief Returns the expected input feature size.
     */
    size_t getInputSize() const override { return m_inputSize; }

    /**
     * @brief Returns the number of output neurons.
     */
    size_t getOutputSize() const override { return m_outputSize; }

    /**
     * @brief Resizes the output neurons and reinitializes weights.
     * @param num New number of output neurons.
     */
    void setNeuronNum(size_t num) override {
        m_outputSize = num;
        m_params = owTensor<float, 1>(m_inputSize * m_outputSize + m_outputSize);
        m_grads = owTensor<float, 1>(m_inputSize * m_outputSize + m_outputSize);
        m_weights = owTensorMap<float, 2>(m_params.data(), owTensorShape{m_inputSize, m_outputSize});
        m_biases = owTensorMap<float, 2>(m_params.data() + (m_inputSize * m_outputSize), owTensorShape{1, m_outputSize});
        m_weightGradients = owTensorMap<float, 2>(m_grads.data(), owTensorShape{m_inputSize, m_outputSize});
        m_biasGradients = owTensorMap<float, 2>(m_grads.data() + (m_inputSize * m_outputSize), owTensorShape{1, m_outputSize});
        initializeWeights();
    }

    /**
     * @brief Initializes weights using Xavier initialization.
     */
    void initializeWeights() {
        std::random_device rd;
        std::default_random_engine generator(rd());
        initializeWeightsWithRNG(generator);
    }

    /**
     * @brief Initializes weights using a specific random number generator.
     */
    template<typename RNG>
    void initializeWeightsWithRNG(RNG& rng) {
        float fanIn = (float)m_inputSize;
        float fanOut = (float)m_outputSize;

        bool isReLU = getActivationName() == "ReLU" || getActivationName() == "LeakyReLU";

        float range;
        if (isReLU) {
            // He (Kaiming) Initialization: optimized for ReLU-like activations
            range = std::sqrt(2.0f / fanIn);
            std::normal_distribution<float> distribution(0.0f, range);
            for (size_t i = 0; i < m_weights.size(); ++i) {
                m_weights.data()[i] = distribution(rng);
            }
        } else {
            // Xavier (Glorot) Initialization: optimized for Sigmoid/Tanh/Identity
            range = std::sqrt(6.0f / (fanIn + fanOut));
            std::uniform_real_distribution<float> distribution(-range, range);
            for (size_t i = 0; i < m_weights.size(); ++i) {
                m_weights.data()[i] = distribution(rng);
            }
        }

        for (size_t i = 0; i < m_biases.size(); ++i) {
            m_biases.data()[i] = 0.0f;
        }
    }

    /**
     * @brief Serializes configuration, weights, and biases to XML.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<InputSize>" << m_inputSize << "</InputSize>\n";
        ss << "<OutputSize>" << m_outputSize << "</OutputSize>\n";
        ss << "<Activation>" << getActivationName() << "</Activation>\n";
        ss << "<Weights>" << m_weights.toString() << "</Weights>\n";
        ss << "<Biases>" << m_biases.toString() << "</Biases>\n";
        ss << "<RegType>" << static_cast<int>(m_regType) << "</RegType>\n";
        ss << "<RegLambda>" << m_regLambda << "</RegLambda>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes configuration, weights, and biases from XML.
     */
    void fromXML(const std::string& xml) override {
        m_inputSize = std::stoul(getTagContent(xml, "InputSize"));
        m_outputSize = std::stoul(getTagContent(xml, "OutputSize"));
        setActivationByName(getTagContent(xml, "Activation"));
        m_params = owTensor<float, 1>(m_inputSize * m_outputSize + m_outputSize);
        m_grads = owTensor<float, 1>(m_inputSize * m_outputSize + m_outputSize);
        m_weights = owTensorMap<float, 2>(m_params.data(), owTensorShape{m_inputSize, m_outputSize});
        m_biases = owTensorMap<float, 2>(m_params.data() + (m_inputSize * m_outputSize), owTensorShape{1, m_outputSize});
        m_weightGradients = owTensorMap<float, 2>(m_grads.data(), owTensorShape{m_inputSize, m_outputSize});
        m_biasGradients = owTensorMap<float, 2>(m_grads.data() + (m_inputSize * m_outputSize), owTensorShape{1, m_outputSize});
        m_weights.fromString(getTagContent(xml, "Weights"));
        m_biases.fromString(getTagContent(xml, "Biases"));
        std::string rt = getTagContent(xml, "RegType");
        if (!rt.empty()) m_regType = static_cast<owRegularizationType>(std::stoi(rt));
        std::string rl = getTagContent(xml, "RegLambda");
        if (!rl.empty()) m_regLambda = std::stof(rl);
    }

    /**
     * @brief Performs forward pass: y = activation(xW + b).
     * @param input Input tensor of shape [Batch, InputSize].
     * @return Output tensor of shape [Batch, OutputSize].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        size_t batchSize = input.shape()[0];
        owTensor<float, 2> z(batchSize, m_outputSize);

#ifdef OW_USE_GPU
        cuda::linearForward(input.data(), m_weights.data(), m_biases.data(), z.data(), 
                           (int)batchSize, (int)m_inputSize, (int)m_outputSize);
#else
        z = input.dot(m_weights);
        for (size_t i = 0; i < batchSize; ++i) {
            for (size_t j = 0; j < m_outputSize; ++j) z(i, j) += m_biases(0, j);
        }
#endif
        m_lastZ = z;
        return m_activation ? m_activation->forward(z) : z;
    }

    /**
     * @brief Performs backward pass: computes gradients for weights, biases, and input.
     * @param outputGradient Gradient from the next layer.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> dz = m_activation ? m_activation->backward(m_lastZ, outputGradient) : outputGradient;
        size_t batchSize = m_lastInput.shape()[0];
        
        m_grads.setZero();

        // 1. Weight Gradients: dW = input^T * dz
        auto inputT = m_lastInput.transpose();
        auto dW = inputT.dot(dz);
        std::copy(dW.data(), dW.data() + dW.size(), m_weightGradients.data());

        // 2. Bias Gradients: db = sum(dz) over batch
        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t j = 0; j < m_outputSize; ++j) m_biasGradients(0, j) += dz(b, j);
        }

        // 3. Input Gradient: dx = dz * W^T
        auto weightsT = m_weights.transpose();
        return dz.dot(weightsT);
    }

    /**
     * @brief Updates weights and biases using the attached optimizer and applies regularization.
     */
    void train() override {
        if (m_optimizer) {
            applyRegularization(m_weights, m_weightGradients);
            m_optimizer->update(m_weights, m_weightGradients);
            m_optimizer->update(m_biases, m_biasGradients);
        }
    }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owLinearLayer>(m_inputSize, m_outputSize);
        copy->m_params = m_params;
        copy->m_layerName = m_layerName;
        if (m_activation) copy->m_activation = m_activation->clone();
        return copy;
    }

	float* getParamsPtr() override { return m_params.data(); }
	float* getGradsPtr() override { return m_grads.data(); }
	size_t getParamsCount() override { return m_params.size(); }

private:
    using owTensorShape = typename owTensor<float, 2>::owTensorShape;
    size_t m_inputSize, m_outputSize;
    owTensor<float, 1> m_params, m_grads;
    owTensorMap<float, 2> m_weights, m_biases, m_weightGradients, m_biasGradients;
    owTensor<float, 2> m_lastInput, m_lastZ;
};

} // namespace ow
