/*
 * owTrendLayer.hpp
 *
 *  Created on: Apr 12, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <vector>
#include <algorithm>

namespace ow {

/**
 * @class owTrendLayer
 * @brief A layer that analyzes and amplifies temporal trends in windowed data.
 * 
 * The Trend Layer is designed for time-series data processed in windows. It identifies
 * the overall "slope" of the window and applies a learnable boost to the most recent 
 * data point. This helps the model capture momentum and directional changes.
 * 
 * Mathematical Operation (Forward):
 * For each sample in the batch:
 * @f$ \text{slope} = x_{t} - x_{0} @f$
 * @f$ y_{t} = x_{t} + (\text{scale} \times \text{slope}) @f$
 * where @f$ x_0 @f$ is the first element, @f$ x_t @f$ is the last element (most recent),
 * and @f$ \text{scale} @f$ is a learnable parameter clamped between 0.0 and 0.5.
 */
class owTrendLayer : public owLayer {
public:
    /**
     * @brief Constructs an owTrendLayer.
     * @param inputSize The number of elements in the input window.
     */
    owTrendLayer(size_t inputSize = 0) : m_size(inputSize), m_params(1, 1), m_grads(1, 1) {
        m_layerName = "Trend Layer";
        m_params(0, 0) = 0.01f; // Safe initial trend sensitivity
    }

    /**
     * @brief Returns the expected input size.
     * @return size_t Input size.
     */
    size_t getInputSize() const override { return m_size; }

    /**
     * @brief Returns the output size (same as input size).
     * @return size_t Output size.
     */
    size_t getOutputSize() const override { return m_size; }

    /**
     * @brief Sets the number of neurons (window size).
     * @param num Number of neurons.
     */
    void setNeuronNum(size_t num) override { m_size = num; }

    /**
     * @brief Forward pass: calculates trend slope and amplifies the last element.
     * 
     * Formula:
     * @f$ \text{output}[win-1] = \text{input}[win-1] + \text{scale} \times (\text{input}[win-1] - \text{input}[0]) @f$
     * 
     * @param input Input tensor of shape [Batch, Window].
     * @return owTensor<float, 2> Processed output tensor.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        owTensor<float, 2> output = input;
        
        size_t batch = input.shape()[0];
        size_t win = input.shape()[1];
        
        if (win < 2) return output;

        // Apply clamped scale to prevent explosion
        float scale = std::max(0.0f, std::min(0.5f, m_params(0, 0)));
        for (size_t i = 0; i < batch; ++i) {
            float slope = input(i, win - 1) - input(i, 0);
            output(i, win - 1) += scale * slope;
        }
        
        return output;
    }

    /**
     * @brief Backward pass: propagates gradients to input and learnable scale.
     * 
     * Mathematical Gradients:
     * Let @f$ L @f$ be the loss, @f$ G @f$ be the output gradient (@f$ \partial L / \partial y @f$):
     * 1. Gradient w.r.t. Scale:
     *    @f$ \frac{\partial L}{\partial \text{scale}} = \sum (G_{win-1} \times (x_{win-1} - x_0)) @f$
     * 2. Gradient w.r.t. Input (last element):
     *    @f$ \frac{\partial L}{\partial x_{win-1}} = G_{win-1} \times (1 + \text{scale}) @f$
     * 3. Gradient w.r.t. Input (first element):
     *    @f$ \frac{\partial L}{\partial x_0} = G_0 - (G_{win-1} \times \text{scale}) @f$
     * 
     * @param outputGradient Gradient of loss w.r.t. output.
     * @return owTensor<float, 2> Gradient of loss w.r.t. input.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> inputGrad = outputGradient;
        
        size_t batch = m_lastInput.shape()[0];
        size_t win = m_lastInput.shape()[1];
        float scale = std::max(0.0f, std::min(0.5f, m_params(0, 0)));
        
        float dScale = 0;
        for (size_t i = 0; i < batch; ++i) {
            float slope = m_lastInput(i, win - 1) - m_lastInput(i, 0);
            dScale += outputGradient(i, win - 1) * slope;
            
            inputGrad(i, win - 1) *= (1.0f + scale);
            inputGrad(i, 0) -= outputGradient(i, win - 1) * scale;
        }
        
        // Manual gradient clipping for the scale parameter
        if (dScale > 1.0f) dScale = 1.0f;
        if (dScale < -1.0f) dScale = -1.0f;

        m_grads(0, 0) += dScale;
        return inputGrad;
    }

    /**
     * @brief Updates the trend sensitivity parameter using the optimizer.
     * Ensures the parameter stays within a safe [0.0, 0.5] range.
     */
    void train() override {
        if (m_optimizer && !m_isFrozen) {
            m_optimizer->update(m_params, m_grads);
            // Force parameter back into safe range after update
            m_params(0, 0) = std::max(0.0f, std::min(0.5f, m_params(0, 0)));
        }
        m_grads.setZero();
    }

    /**
     * @brief Returns a pointer to the parameter data (scale).
     * @return float* Pointer to parameters.
     */
    float* getParamsPtr() override { return m_params.data(); }

    /**
     * @brief Returns a pointer to the gradient data.
     * @return float* Pointer to gradients.
     */
    float* getGradsPtr() override { return m_grads.data(); }

    /**
     * @brief Returns the total number of learnable parameters (1).
     * @return size_t Parameter count.
     */
    size_t getParamsCount() override { return 1; }

    /**
     * @brief Creates a deep copy of the layer.
     * @return std::shared_ptr<owLayer> Cloned layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owTrendLayer>(m_size);
        copy->m_params(0, 0) = m_params(0, 0);
        return copy;
    }

    /**
     * @brief Serializes the layer state to XML format.
     * @return std::string XML representation.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<Size>" << m_size << "</Size>\n";
        ss << "<SlopeScale>" << m_params(0, 0) << "</SlopeScale>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes the layer state from XML format.
     * @param xml XML string.
     */
    void fromXML(const std::string& xml) override {
        m_size = std::stoul(getTagContent(xml, "Size"));
        m_params(0, 0) = std::stof(getTagContent(xml, "SlopeScale"));
    }

private:
    size_t m_size;
    owTensor<float, 2> m_params, m_grads;
    owTensor<float, 2> m_lastInput;
};

} // namespace ow
