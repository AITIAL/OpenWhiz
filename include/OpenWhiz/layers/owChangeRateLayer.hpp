/*
 * owChangeRateLayer.hpp
 *
 *  Created on: Apr 15, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <vector>

namespace ow {

/**
 * @class owChangeRateLayer
 * @brief A layer that transforms absolute values into percentage change rates.
 * 
 * This layer is particularly useful for non-stationary time series (like stock prices 
 * or currency rates) where the absolute value grows over time but the relative 
 * changes remain within a stable range.
 * 
 * Mathematical Operation (Forward):
 * For an input window @f$ X = [x_0, x_1, \dots, x_{n-1}] @f$, the output is:
 * @f$ y_t = \frac{x_t - x_{t-1}}{x_{t-1}} @f$ for @f$ t = 1 \dots n-1 @f$.
 * 
 * The output size will be @f$ n-1 @f$.
 */
class owChangeRateLayer : public owLayer {
public:
    /**
     * @brief Constructs an owChangeRateLayer.
     * @param inputSize The number of elements in the input window.
     */
    owChangeRateLayer(size_t inputSize = 0) : m_inputSize(inputSize) {
        m_layerName = "Change Rate Layer";
    }

    /**
     * @brief Returns the expected input size.
     * @return size_t Input size.
     */
    size_t getInputSize() const override { return m_inputSize; }

    /**
     * @brief Returns the output size (InputSize - 1).
     * @return size_t Output size.
     */
    size_t getOutputSize() const override { return m_inputSize > 1 ? m_inputSize - 1 : 0; }

    /**
     * @brief Sets the number of neurons (input window size).
     * @param num Number of neurons.
     */
    void setNeuronNum(size_t num) override { m_inputSize = num; }

    /**
     * @brief Forward pass: calculates percentage changes between consecutive elements.
     * 
     * Formula: @f$ y_{i} = \frac{x_{i+1} - x_i}{x_i} @f$
     * 
     * @param input Input tensor of shape [Batch, InputSize].
     * @return owTensor<float, 2> Output tensor of shape [Batch, InputSize - 1].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        size_t batch = input.shape()[0];
        size_t win = input.shape()[1];
        
        if (win < 2) return owTensor<float, 2>(batch, 0);

        owTensor<float, 2> output(batch, win - 1);
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < win - 1; ++j) {
                float x_prev = input(i, j);
                float x_curr = input(i, j + 1);
                
                // Avoid division by zero
                if (std::abs(x_prev) < 1e-9f) {
                    output(i, j) = 0.0f;
                } else {
                    output(i, j) = (x_curr - x_prev) / x_prev;
                }
            }
        }
        return output;
    }

    /**
     * @brief Backward pass: propagates gradients to the input values.
     * 
     * Mathematical Gradients:
     * Let @f$ y_j = \frac{x_{j+1} - x_j}{x_j} = \frac{x_{j+1}}{x_j} - 1 @f$.
     * Let @f$ G_j @f$ be the output gradient @f$ \partial L / \partial y_j @f$.
     * 
     * 1. Gradient w.r.t. @f$ x_{j+1} @f$ (current element):
     *    @f$ \frac{\partial L}{\partial x_{j+1}} = \frac{G_j}{x_j} @f$
     * 
     * 2. Gradient w.r.t. @f$ x_j @f$ (previous element):
     *    @f$ \frac{\partial L}{\partial x_j} = -G_j \times \frac{x_{j+1}}{x_j^2} @f$
     * 
     * The total gradient for an intermediate @f$ x_j @f$ is the sum of contributions 
     * from @f$ y_{j-1} @f$ and @f$ y_j @f$.
     * 
     * @param outputGradient Gradient of loss w.r.t. output.
     * @return owTensor<float, 2> Gradient of loss w.r.t. input.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batch = m_lastInput.shape()[0];
        size_t win = m_lastInput.shape()[1];
        
        owTensor<float, 2> inputGrad(batch, win);
        inputGrad.setZero();

        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < win - 1; ++j) {
                float x_prev = m_lastInput(i, j);
                float x_curr = m_lastInput(i, j + 1);
                float G = outputGradient(i, j);

                if (std::abs(x_prev) > 1e-9f) {
                    // Contribution to x_{j+1}
                    inputGrad(i, j + 1) += G / x_prev;
                    // Contribution to x_j
                    inputGrad(i, j) -= G * (x_curr / (x_prev * x_prev));
                }
            }
        }
        return inputGrad;
    }

    /**
     * @brief No learnable parameters in this layer.
     */
    void train() override {}

    float* getParamsPtr() override { return nullptr; }
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override { return 0; }

    /**
     * @brief Creates a deep copy of the layer.
     * @return std::shared_ptr<owLayer> Cloned layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        return std::make_shared<owChangeRateLayer>(m_inputSize);
    }

    /**
     * @brief Serializes the layer configuration to XML.
     * @return std::string XML representation.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<InputSize>" << m_inputSize << "</InputSize>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes the layer configuration from XML.
     * @param xml XML string.
     */
    void fromXML(const std::string& xml) override {
        std::string content = getTagContent(xml, "InputSize");
        if (!content.empty()) m_inputSize = std::stoul(content);
    }

private:
    size_t m_inputSize;
    owTensor<float, 2> m_lastInput;
};

} // namespace ow
