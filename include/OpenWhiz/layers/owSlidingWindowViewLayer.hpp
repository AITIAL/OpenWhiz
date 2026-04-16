/*
 * owSlidingWindowViewLayer.hpp
 *
 *  Created on: Apr 16, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"

namespace ow {

/**
 * @class owSlidingWindowViewLayer
 * @brief A stateless layer that slices temporal windows from a pre-formatted forecasting dataset.
 * 
 * This is the "View" version of the sliding window layer. It expects the dataset to have 
 * already prepared the master history (via owDataset::prepareForecastData).
 */
class owSlidingWindowViewLayer : public owLayer {
public:
    owSlidingWindowViewLayer(size_t windowSize = 5, size_t dilation = 1, size_t masterWindowSize = 5, bool includeCurrent = true) 
        : m_windowSize(windowSize), m_dilation(dilation), m_masterWindowSize(masterWindowSize), m_inputFeatures(1), m_includeCurrent(includeCurrent) {
        m_layerName = "Sliding Window View Layer";
    }

    size_t getInputSize() const override { return m_masterWindowSize + m_inputFeatures; } 
    size_t getOutputSize() const override { return m_windowSize + (m_includeCurrent ? m_inputFeatures : 0); }
    
    void setNeuronNum(size_t num) override { m_inputFeatures = num; }
    void reset() override {}

    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owSlidingWindowViewLayer>(m_windowSize, m_dilation, m_masterWindowSize, m_includeCurrent);
        copy->m_layerName = m_layerName;
        copy->m_inputFeatures = m_inputFeatures;
        return copy;
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batchSize = input.shape()[0];
        size_t totalCols = input.shape()[1];
        m_inputFeatures = totalCols - m_masterWindowSize; 

        owTensor<float, 2> output(batchSize, getOutputSize());
        
        for (size_t i = 0; i < batchSize; ++i) {
            for (size_t w = 0; w < m_windowSize; ++w) {
                size_t masterIdx = m_masterWindowSize - m_windowSize + w;
                output(i, w) = input(i, masterIdx);
            }
            if (m_includeCurrent) {
                for (size_t f = 0; f < m_inputFeatures; ++f) {
                    output(i, m_windowSize + f) = input(i, m_masterWindowSize + f);
                }
            }
        }
        return output;
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batchSize = outputGradient.shape()[0];
        owTensor<float, 2> inputGradient(batchSize, m_masterWindowSize + m_inputFeatures);
        
        for (size_t i = 0; i < batchSize; ++i) {
            for (size_t w = 0; w < m_windowSize; ++w) {
                size_t masterIdx = m_masterWindowSize - m_windowSize + w;
                inputGradient(i, masterIdx) = outputGradient(i, w);
            }
            if (m_includeCurrent) {
                for (size_t f = 0; f < m_inputFeatures; ++f) {
                    inputGradient(i, m_masterWindowSize + f) = outputGradient(i, m_windowSize + f);
                }
            }
        }
        return inputGradient;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<WindowSize>" << m_windowSize << "</WindowSize>\n";
        ss << "<Dilation>" << m_dilation << "</Dilation>\n";
        ss << "<MasterWindowSize>" << m_masterWindowSize << "</MasterWindowSize>\n";
        ss << "<InputFeatures>" << m_inputFeatures << "</InputFeatures>\n";
        ss << "<IncludeCurrent>" << (m_includeCurrent ? 1 : 0) << "</IncludeCurrent>\n";
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        m_windowSize = std::stoul(getTagContent(xml, "WindowSize"));
        m_dilation = std::stoul(getTagContent(xml, "Dilation"));
        m_masterWindowSize = std::stoul(getTagContent(xml, "MasterWindowSize"));
        m_inputFeatures = std::stoul(getTagContent(xml, "InputFeatures"));
        m_includeCurrent = std::stoi(getTagContent(xml, "IncludeCurrent")) != 0;
    }

    void train() override {}
    float* getParamsPtr() override { return nullptr; }
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override { return 0; }

    void setIncludeCurrent(bool include) { m_includeCurrent = include; }
    void setMasterWindowSize(size_t size) { m_masterWindowSize = size; }

private:
    size_t m_windowSize;    
    size_t m_dilation;      
    size_t m_masterWindowSize; 
    size_t m_inputFeatures; 
    bool m_includeCurrent;  
};

} // namespace ow
