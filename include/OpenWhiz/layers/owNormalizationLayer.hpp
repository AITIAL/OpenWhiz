/*
 * owNormalizationLayer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include "../core/owNeuralNetwork.hpp"

namespace ow {

/**
 * @class owNormalizationLayer
 * @brief Performs Min-Max normalization on input features.
 * 
 * This layer scales each feature of the input to a range between 0 and 1 (or any range based on 
 * provided statistics). It is typically used at the beginning of a network to ensure all input 
 * features have a similar scale, which improves convergence.
 * 
 * @details
 * **Implementation Details:**
 * - Normalization formula: x_norm = (x - min) / (max - min).
 * - Handles cases where (max - min) is near zero by defaulting the divisor to 1.0 to prevent 
 *   division by zero.
 * - Statistics (min, max) can be provided via `setStatistics` or automatically retrieved 
 *   from the parent network's dataset.
 * 
 * **Unique Features:**
 * - Deterministic scaling based on pre-calculated dataset statistics.
 * - Supports both forward normalization and gradient backpropagation.
 * - **Automatic Configuration:** If size is not provided at construction, it will auto-configure 
 *   using dataset statistics when the first forward pass occurs or when added to a network.
 * 
 * **Platform-Specific Notes:**
 * - **Computer/Mobile/Web:** Extremely lightweight; involves only basic arithmetic operations.
 * - **Industrial:** Essential for sensor data where different sensors (e.g., temperature vs. pressure) 
 *   have vastly different units and scales.
 */
class owNormalizationLayer : public owLayer {
public:
    /**
     * @brief Constructs a Normalization layer.
     * @param size The number of features in the input (optional, can be auto-detected).
     */
    owNormalizationLayer(size_t size = 0) : m_size(size), m_min(1, size), m_max(1, size) {
        m_layerName = "Normalization Layer";
        if (m_size > 0) {
            m_min.setZero(); m_max.setConstant(1.0f);
        }
    }
    size_t getInputSize() const override { return m_size; }
    size_t getOutputSize() const override { return m_size; }
    void setInputSize(size_t size) override { 
        setNeuronNum(size); 
        autoConfigure(); // Fetch real statistics immediately
    }
    void setNeuronNum(size_t num) override {
        m_size = num; m_min = owTensor<float, 2>(1, m_size); m_max = owTensor<float, 2>(1, m_size);
        m_min.setZero(); m_max.setConstant(1.0f);
    }
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owNormalizationLayer>(m_size);
        copy->m_min = m_min; copy->m_max = m_max; copy->m_layerName = m_layerName;
        copy->m_parentNetwork = m_parentNetwork;
        return copy;
    }
    std::string toXML() const override {
        std::stringstream ss; ss << "<Size>" << m_size << "</Size>\n<Min>" << m_min.toString() << "</Min>\n<Max>" << m_max.toString() << "</Max>\n";
        return ss.str();
    }
    void fromXML(const std::string& xml) override {
        m_size = std::stoul(getTagContent(xml, "Size"));
        m_min = owTensor<float, 2>(1, m_size); m_max = owTensor<float, 2>(1, m_size);
        m_min.fromString(getTagContent(xml, "Min")); m_max.fromString(getTagContent(xml, "Max"));
    }
    void setStatistics(const owTensor<float, 2>& min, const owTensor<float, 2>& max) { 
        m_min = min; m_max = max; m_size = min.shape()[1];
    }
    
    /**
     * @brief Automatically fetches statistics from the parent neural network's dataset.
     */
    void autoConfigure() {
        if (m_parentNetwork) {
            m_parentNetwork->getInputMinMax(m_min, m_max);
            m_size = m_min.shape()[1];
        }
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        if (m_size == 0 || m_min.size() == 0) autoConfigure();
        if (m_size == 0) return input; // Safety fallback

        owTensor<float, 2> output(input.shape());
        for (size_t b = 0; b < input.shape()[0]; ++b) {
            for (size_t f = 0; f < m_size; ++f) {
                float range = m_max(0, f) - m_min(0, f);
                output(b, f) = (input(b, f) - m_min(0, f)) / (std::abs(range) < 1e-7f ? 1.0f : range);
            }
        }
        return output;
    }
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        if (m_size == 0) return outputGradient;

        owTensor<float, 2> inputGradient(outputGradient.shape());
        for (size_t b = 0; b < outputGradient.shape()[0]; ++b) {
            for (size_t f = 0; f < m_size; ++f) {
                float range = m_max(0, f) - m_min(0, f);
                inputGradient(b, f) = outputGradient(b, f) / (std::abs(range) < 1e-7f ? 1.0f : range);
            }
        }
        return inputGradient;
    }
    void train() override {}

	float* getParamsPtr() override { return nullptr; }
	float* getGradsPtr() override { return nullptr; }
	size_t getParamsCount() override { return 0; }

private:
    size_t m_size; owTensor<float, 2> m_min, m_max;
};

} // namespace ow
