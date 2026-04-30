/*
 * owCacheLayer.hpp
 *
 *  Created on: Apr 16, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include "../core/owNeuralNetwork.hpp"
#include "../data/owDataset.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>

namespace ow {

/**
 * @class owCacheLayer
 * @brief Performance optimizer that records pre-processed data for the ENTIRE dataset.
 * 
 * This layer is designed to be placed after non-trainable preprocessing layers 
 * (like Normalization or Sliding Window). During the first pass, it records the outputs. 
 * In subsequent passes, it returns the cached tensors, bypassing the calculation 
 * of previous layers and significantly speeding up training.
 * 
 * **Unique Features:**
 * - **Subset Awareness:** Automatically detects if it's being asked for Training, 
 *   Validation, or Test subsets and filters its cache accordingly.
 * - **Smart Shuffling:** If enabled, it shuffles the Training subset to improve 
 *   generalization without losing alignment with targets.
 */
class owCacheLayer : public owLayer {
public:
    owCacheLayer(bool shuffle = true) 
        : m_shuffleEnabled(shuffle), m_isFull(false), m_playbackMode(false) {
        m_layerName = "Cache Layer";
    }

    size_t getInputSize() const override { return m_inputDim; }
    size_t getOutputSize() const override { return m_inputDim; }
    void setInputSize(size_t size) override { m_inputDim = size; }
    void setNeuronNum(size_t num) override { m_inputDim = num; }

    /**
     * @brief Resets the playback indices. If shuffling is enabled, it re-shuffles the training set.
     */
    void reset() override {
        if (m_isFull && m_shuffleEnabled && m_isTraining) {
            std::shuffle(m_trainingIndices.begin(), m_trainingIndices.end(), m_rng);
        }
    }

    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owCacheLayer>(m_shuffleEnabled);
        copy->m_layerName = m_layerName;
        copy->m_inputDim = m_inputDim;
        copy->m_parentNetwork = m_parentNetwork;
        return copy;
    }

    /**
     * @brief Core logic for recording and playback.
     * 
     * Handles recording of full dataset and subsequent playback of subsets.
     * Correctly handles row filtering to prevent dimension mismatches in optimizers.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        if (!m_isFull && m_isTraining) {
            // Recording Pass (typically one forward pass with the entire dataset)
            if (m_cachedInputs.size() > 0 && input.shape()[0] > m_cachedInputs[0].shape()[0]) {
                m_cachedInputs.clear();
                m_cachedTargets.clear();
            }
            m_cachedInputs.push_back(input);
            if (m_localTarget) m_cachedTargets.push_back(*m_localTarget); 
            m_inputDim = input.shape()[1];
            // std::cout << "CacheLayer: Recorded " << input.shape()[0] << " rows." << std::endl;
            return input;
        } else if (m_playbackMode) {
            if (m_cachedInputs.empty()) return input;
            
            const auto& fullIn = m_cachedInputs[0];
            const auto& fullOut = m_cachedTargets.empty() ? owTensor<float, 2>() : m_cachedTargets[0];
            size_t inputRows = input.shape()[0];
            size_t totalRows = fullIn.shape()[0];

            // 1. Full Dataset Playback
            if (inputRows == totalRows) {
                m_activeTarget = fullOut;
                return fullIn;
            }

            // 2. Subset Playback (Training / Validation / Test)
            std::vector<size_t>* targetIndices = nullptr;
            if (inputRows == m_trainingIndices.size()) targetIndices = &m_trainingIndices;
            else if (inputRows == m_validationIndices.size()) targetIndices = &m_validationIndices;
            else if (inputRows == m_testIndices.size()) targetIndices = &m_testIndices;

            if (targetIndices) {
                owTensor<float, 2> subsetIn(inputRows, fullIn.shape()[1]);
                owTensor<float, 2> subsetOut(inputRows, fullOut.shape()[1]);
                
                for (size_t i = 0; i < inputRows; ++i) {
                    size_t idx = (*targetIndices)[i];
                    if (idx >= totalRows) continue; // Safety
                    for (size_t j = 0; j < fullIn.shape()[1]; ++j) subsetIn(i, j) = fullIn(idx, j);
                    if (fullOut.size() > 0) {
                        for (size_t j = 0; j < fullOut.shape()[1]; ++j) subsetOut(i, j) = fullOut(idx, j);
                    }
                }
                m_activeTarget = subsetOut;
                return subsetIn;
            }

            return input; 
        }
        return input;
    }

    const owTensor<float, 2>& getActiveTarget() const override {
        return m_activeTarget;
    }

    /**
     * @brief Transitions the layer from recording to playback.
     * Pre-calculates indices for Training, Validation, and Test subsets.
     */
    void lockCache() override {
        if (m_cachedInputs.empty()) return;
        m_isFull = true;
        m_playbackMode = true; 

        if (m_parentNetwork && m_parentNetwork->getDataset()) {
            auto* ds = m_parentNetwork->getDataset();
            const auto& types = ds->getSampleTypes();
            
            m_trainingIndices.clear();
            m_validationIndices.clear();
            m_testIndices.clear();

            for (size_t i = 0; i < types.size(); ++i) {
                if (types[i] == SampleType::Training) m_trainingIndices.push_back(i);
                else if (types[i] == SampleType::Validation) m_validationIndices.push_back(i);
                else if (types[i] == SampleType::Test) m_testIndices.push_back(i);
            }
        }
    }

    void setPlaybackMode(bool enabled) override {
        m_playbackMode = enabled;
    }

    /**
     * @brief Exports the cached data to a CSV file with smart Header naming.
     */
    void saveToCSV(const std::string& filepath, 
                   const std::vector<std::string>& dates = {},
                   const std::vector<float>& targets = {}) const {
        if (m_cachedInputs.empty()) return;

        std::ofstream file(filepath);
        if (!file.is_open()) return;

        char delim = ';';
        std::vector<std::string> inputColNames;
        std::vector<std::string> targetColNames;

        if (m_parentNetwork && m_parentNetwork->getDataset()) {
            auto* ds = m_parentNetwork->getDataset();
            delim = ds->getDelimiter();
            
            auto inIndices = ds->getUsedColumnIndices(false);
            auto outIndices = ds->getUsedColumnIndices(true);
            
            for(int idx : inIndices) inputColNames.push_back(ds->getColumnName(idx));
            for(int idx : outIndices) targetColNames.push_back(ds->getColumnName(idx));
        }

        owTensor<float, 2> tMin, tMax;
        bool useTargetNorm = false;
        if (m_parentNetwork) {
            for (auto& l : m_parentNetwork->getLayers()) {
                if (l->getLayerName() == "Inverse Normalization Layer" && l->isEnabled()) {
                    useTargetNorm = true;
                    m_parentNetwork->getTargetMinMax(tMin, tMax);
                    break;
                }
            }
        }

        const auto& inputBatch = m_cachedInputs[0];
        const auto& targetBatch = m_cachedTargets.empty() ? owTensor<float, 2>() : m_cachedTargets[0];
        
        // 1. Write Header Row with Lag suffixes
        if (!dates.empty()) file << "Date" << delim;
        
        size_t totalInputFeatures = inputBatch.shape()[1];
        size_t originalInputCount = inputColNames.empty() ? 1 : inputColNames.size();
        size_t windowSize = totalInputFeatures / originalInputCount;

        for (size_t j = 0; j < totalInputFeatures; ++j) {
            size_t origIdx = j % originalInputCount;
            size_t lag = windowSize - (j / originalInputCount);
            std::string baseName = inputColNames.empty() ? "Input" : inputColNames[origIdx];
            file << baseName << "_lag" << lag << delim;
        }

        size_t tCount = targetBatch.size() > 0 ? targetBatch.shape()[1] : (targets.empty() ? 0 : 1);
        for (size_t j = 0; j < tCount; ++j) {
            std::string baseName = targetColNames.empty() ? "Target" : targetColNames[j];
            file << baseName << (j == tCount - 1 ? "" : std::string(1, delim));
        }
        file << "\n";

        // 2. Write Data Rows
        size_t globalRowIdx = 0;
        size_t rows = inputBatch.shape()[0];
        bool recordingStarted = false;

        for (size_t r = 0; r < rows; ++r) {
            if (!recordingStarted) {
                if (std::abs(inputBatch(r, 0)) > 1e-7f) {
                    recordingStarted = true;
                } else {
                    globalRowIdx++;
                    continue;
                }
            }

            if (!dates.empty() && globalRowIdx < dates.size()) {
                file << dates[globalRowIdx] << delim;
            }

            for (size_t j = 0; j < inputBatch.shape()[1]; ++j) {
                file << inputBatch(r, j) << delim;
            }

            if (!targets.empty() && globalRowIdx < targets.size()) {
                float val = targets[globalRowIdx];
                if (useTargetNorm && tMin.size() > 0) {
                    float range = tMax(0, 0) - tMin(0, 0);
                    val = (val - tMin(0, 0)) / (std::abs(range) < 1e-7f ? 1.0f : range);
                }
                file << val;
            } else if (targetBatch.size() > 0 && r < targetBatch.shape()[0]) {
                for (size_t j = 0; j < targetBatch.shape()[1]; ++j) {
                    float val = targetBatch(r, j);
                    if (useTargetNorm && tMin.size() > (size_t)j) {
                        float range = tMax(0, j) - tMin(0, j);
                        val = (val - tMin(0, j)) / (std::abs(range) < 1e-7f ? 1.0f : range);
                    }
                    file << val << (j == targetBatch.shape()[1] - 1 ? "" : std::string(1, delim));
                }
            }
            file << "\n";
            globalRowIdx++;
        }
        file.close();
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        return outputGradient;
    }

    void train() override {}
    float* getParamsPtr() override { return nullptr; }
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override { return 0; }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<ShuffleEnabled>" << (m_shuffleEnabled ? 1 : 0) << "</ShuffleEnabled>\n";
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        m_shuffleEnabled = std::stoi(getTagContent(xml, "ShuffleEnabled")) != 0;
    }

    bool isFull() const override { return m_isFull; }
    void setFull(bool full) { m_isFull = full; }

private:
    bool m_shuffleEnabled;
    bool m_isFull;
    bool m_playbackMode;
    size_t m_inputDim = 0;
    
    std::vector<owTensor<float, 2>> m_cachedInputs;
    std::vector<owTensor<float, 2>> m_cachedTargets;
    
    std::vector<size_t> m_trainingIndices;
    std::vector<size_t> m_validationIndices;
    std::vector<size_t> m_testIndices;

    owTensor<float, 2> m_activeTarget;

    std::mt19937 m_rng{std::random_device{}()};
};

} // namespace ow
