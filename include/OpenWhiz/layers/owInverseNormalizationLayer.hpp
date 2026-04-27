/*
 * owInverseNormalizationLayer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include "../core/owNeuralNetwork.hpp"

namespace ow {

/**
 * @class owInverseNormalizationLayer
 * @brief Maps normalized data back to its original scale and range.
 * 
 * The owInverseNormalizationLayer is typically used as a post-processing step
 * to convert the model's normalized output (usually in range [0, 1]) back into
 * meaningful physical units (e.g., Temperature, Pressure, Speed).
 * 
 * Implementation Details:
 * - Forward: output = input * (max - min) + min.
 * - This layer can automatically fetch target statistics from the parent network's dataset.
 * 
 * Platform Notes:
 * - Industrial: Crucial for displaying real-world values in HMI (Human-Machine Interface)
 *   systems after the model has processed them.
 * - Computer/Mobile: Highly efficient element-wise operation.
 */
class owInverseNormalizationLayer : public owLayer {
public:
    /**
     * @brief Constructor for owInverseNormalizationLayer.
     * @param size Number of features to inverse normalize (optional, can be auto-detected).
     */
    owInverseNormalizationLayer(size_t size = 0) : m_size(size), m_min(1, size), m_max(1, size) {
        m_layerName = "Inverse Normalization Layer";
        if (m_size > 0) {
            m_min.setZero(); m_max.setConstant(1.0f);
        }
    }

    /**
     * @brief Returns the expected input feature size.
     */
    size_t getInputSize() const override { return m_size; }

    /**
     * @brief Returns the output feature size.
     */
    size_t getOutputSize() const override { return m_size; }

    /** @brief Sets the input feature size. */
    void setInputSize(size_t size) override { 
        setNeuronNum(size); 
        autoConfigure(); // Fetch real statistics immediately
    }

    /**
     * @brief Resizes the layer and reinitializes statistics.
     * @param num New feature size.
     */
    void setNeuronNum(size_t num) override {
        m_size = num; m_min = owTensor<float, 2>(1, m_size); m_max = owTensor<float, 2>(1, m_size);
        m_min.setZero(); m_max.setConstant(1.0f);
    }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owInverseNormalizationLayer>(m_size);
        copy->m_min = m_min; copy->m_max = m_max; copy->m_layerName = m_layerName;
        copy->m_parentNetwork = m_parentNetwork;
        return copy;
    }

    /**
     * @brief Serializes configuration and statistics to XML.
     */
    std::string toXML() const override {
        std::stringstream ss; ss << "<Size>" << m_size << "</Size>\n<Min>" << m_min.toString() << "</Min>\n<Max>" << m_max.toString() << "</Max>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes configuration and statistics from XML.
     */
    void fromXML(const std::string& xml) override {
        m_size = std::stoul(getTagContent(xml, "Size"));
        m_min = owTensor<float, 2>(1, m_size); m_max = owTensor<float, 2>(1, m_size);
        m_min.fromString(getTagContent(xml, "Min")); m_max.fromString(getTagContent(xml, "Max"));
    }

    /**
     * @brief Sets the min/max statistics for each feature.
     * @param min 1xN tensor of minimum values.
     * @param max 1xN tensor of maximum values.
     */
    void setStatistics(const owTensor<float, 2>& min, const owTensor<float, 2>& max) { 
        m_min = min; m_max = max; m_size = min.shape()[1];
    }

    /**
     * @brief Automatically fetches statistics from the parent neural network's dataset.
     */
    void autoConfigure() {
        if (m_parentNetwork) {
            m_parentNetwork->getTargetMinMax(m_min, m_max);
            m_size = m_min.shape()[1];
        }
    }

    /**
     * @brief Performs forward pass: maps [0, 1] range to [min, max].
     * @param input Normalized input tensor.
     * @return Denormalized output tensor.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        if (!m_enabled) return input; // Pass-through if disabled
        if (m_size == 0 || m_min.size() == 0) autoConfigure();
        if (m_size == 0) return input;

        owTensor<float, 2> output(input.shape());
        for (size_t b = 0; b < input.shape()[0]; ++b) {
            for (size_t f = 0; f < m_size; ++f) output(b, f) = input(b, f) * (m_max(0, f) - m_min(0, f)) + m_min(0, f);
        }
        return output;
    }

    /**
     * @brief Performs backward pass: scales gradients by (max - min).
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        if (m_size == 0) return outputGradient;

        owTensor<float, 2> inputGradient(outputGradient.shape());
        for (size_t b = 0; b < outputGradient.shape()[0]; ++b) {
            for (size_t f = 0; f < m_size; ++f) inputGradient(b, f) = outputGradient(b, f) * (m_max(0, f) - m_min(0, f));
        }
        return inputGradient;
    }

    /**
     * @brief Training step (no-op).
     */
    void train() override {}

	float* getParamsPtr() override { return nullptr; }
	float* getGradsPtr() override { return nullptr; }
	size_t getParamsCount() override { return 0; }

private:
    size_t m_size; owTensor<float, 2> m_min, m_max;
};

} // namespace ow

