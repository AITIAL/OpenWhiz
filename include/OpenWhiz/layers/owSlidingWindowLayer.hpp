/*
 * owSlidingWindowLayer.hpp
 *
 *  Created on: Apr 16, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <deque>

namespace ow {

/**
 * @class owSlidingWindowLayer
 * @brief A stateful layer that maintains an internal buffer to create temporal windows.
 * 
 * Unlike the View version, this layer can work with raw sequential data. It buffers 
 * incoming samples and produces a window of size `windowSize`.
 * 
 * **Usage:** Place it at the beginning of the network to handle time-series logic 
 * without modifying the dataset.
 */
class owSlidingWindowLayer : public owLayer {
public:
    owSlidingWindowLayer(size_t windowSize = 5, size_t dilation = 1, bool includeCurrent = true) 
        : m_windowSize(windowSize), m_dilation(dilation), m_includeCurrent(includeCurrent), m_inputFeatures(0) {
        m_layerName = "Sliding Window Layer";
    }

    size_t getInputSize() const override { return m_inputFeatures; } 
    size_t getOutputSize() const override { return m_windowSize + (m_includeCurrent ? m_inputFeatures : 0); }
    
    void setInputSize(size_t size) override { m_inputFeatures = size; }
    void setNeuronNum(size_t num) override { m_inputFeatures = num; }

    /** @brief Resets the internal buffer. Important between epochs or different sequences. */
    void reset() override { m_buffer.clear(); }

    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owSlidingWindowLayer>(m_windowSize, m_dilation, m_includeCurrent);
        copy->m_layerName = m_layerName;
        copy->m_inputFeatures = m_inputFeatures;
        return copy;
    }

    /**
     * @brief Processes sequential input and returns a windowed output.
     * Note: In batch mode, it treats each row as a consecutive time step.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batchSize = input.shape()[0];
        m_inputFeatures = input.shape()[1];

        owTensor<float, 2> output(batchSize, getOutputSize());
        
        for (size_t i = 0; i < batchSize; ++i) {
            // Fill History using ALREADY buffered samples (T-1, T-2...)
            // This prevents using the current input T to predict current target T
            for (size_t w = 0; w < m_windowSize; ++w) {
                if (m_buffer.size() >= (m_windowSize - w) * m_dilation) {
                    // We want chronological order: oldest to newest
                    // Buffer contains samples up to T-1
                    size_t bufferIdx = m_buffer.size() - (m_windowSize - w) * m_dilation;
                    output(i, w) = m_buffer[bufferIdx];
                } else {
                    output(i, w) = 0.0f;
                }
            }

            // Append Current Features if requested (non-target features)
            if (m_includeCurrent) {
                for (size_t f = 0; f < m_inputFeatures; ++f) {
                    output(i, m_windowSize + f) = input(i, f);
                }
            }

            // Push current sample to buffer for use in FUTURE time steps (T+1, T+2...)
            m_buffer.push_back(input(i, 0));
            if (m_buffer.size() > m_windowSize * m_dilation) {
                m_buffer.pop_front();
            }
        }
        return output;
    }

    /** @brief Pass-through gradients for the current features. */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batchSize = outputGradient.shape()[0];
        owTensor<float, 2> inputGradient(batchSize, m_inputFeatures);
        
        if (m_includeCurrent) {
            for (size_t i = 0; i < batchSize; ++i) {
                for (size_t f = 0; f < m_inputFeatures; ++f) {
                    inputGradient(i, f) = outputGradient(i, m_windowSize + f);
                }
            }
        }
        return inputGradient;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<WindowSize>" << m_windowSize << "</WindowSize>\n";
        ss << "<Dilation>" << m_dilation << "</Dilation>\n";
        ss << "<InputFeatures>" << m_inputFeatures << "</InputFeatures>\n";
        ss << "<IncludeCurrent>" << (m_includeCurrent ? 1 : 0) << "</IncludeCurrent>\n";
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        m_windowSize = std::stoul(getTagContent(xml, "WindowSize"));
        m_dilation = std::stoul(getTagContent(xml, "Dilation"));
        m_inputFeatures = std::stoul(getTagContent(xml, "InputFeatures"));
        m_includeCurrent = std::stoi(getTagContent(xml, "IncludeCurrent")) != 0;
    }

    void train() override {}
    float* getParamsPtr() override { return nullptr; }
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override { return 0; }

private:
    size_t m_windowSize;    
    size_t m_dilation;      
    bool m_includeCurrent;
    size_t m_inputFeatures;
    std::deque<float> m_buffer;
};

} // namespace ow
