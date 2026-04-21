/*
 * owCacheLayer.hpp
 *
 *  Created on: Apr 16, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

namespace ow {

/**
 * @class owCacheLayer
 * @brief Records pre-processed data during the first epoch and replays it later.
 * 
 * This layer is designed to be placed after non-trainable preprocessing layers 
 * (like Normalization or SlidingWindow). It caches the output of those layers 
 * and their corresponding targets.
 * 
 * **Shuffle:** If enabled, this layer shuffles its internal cache after the 
 * first epoch, providing a performance-optimized shuffle mechanism.
 */
class owCacheLayer : public owLayer {
public:
    owCacheLayer(bool shuffle = true) 
        : m_shuffleEnabled(shuffle), m_isFull(false), m_currentBatchIdx(0) {
        m_layerName = "Cache Layer";
    }

    size_t getInputSize() const override { return m_inputDim; }
    size_t getOutputSize() const override { return m_inputDim; }
    void setInputSize(size_t size) override { m_inputDim = size; }
    void setNeuronNum(size_t num) override { m_inputDim = num; }

    void reset() override {
        // We don't clear the cache on reset, only the playback pointer
        m_currentBatchIdx = 0;
        if (m_isFull && m_shuffleEnabled) {
            std::shuffle(m_indices.begin(), m_indices.end(), m_rng);
        }
    }

    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owCacheLayer>(m_shuffleEnabled);
        copy->m_layerName = m_layerName;
        copy->m_inputDim = m_inputDim;
        return copy;
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        if (!m_isFull) {
            // --- RECORDING MODE ---
            // Store Input
            m_cachedInputs.push_back(input);
            
            // Store Target (if available from owLayer::m_localTarget)
            if (m_localTarget) {
                // localTarget contains the whole dataset target, 
                // but we need to capture the exact rows corresponding to this input batch.
                // Since this is the first epoch, we assume sequential access.
                m_cachedTargets.push_back(*m_localTarget); 
            }

            m_inputDim = input.shape()[1];
            return input;
        } else {
            // --- PLAYBACK MODE ---
            size_t idx = m_indices[m_currentBatchIdx];
            m_currentBatchIdx = (m_currentBatchIdx + 1) % m_cachedInputs.size();

            // Return cached input
            return m_cachedInputs[idx];
        }
    }

    /**
     * @brief In Playback mode, returns the target corresponding to the current shuffled batch.
     */
    const owTensor<float, 2>& getActiveTarget() const {
        if (!m_isFull || m_cachedTargets.empty()) {
            static owTensor<float, 2> empty;
            return empty;
        }
        size_t lastIdx = (m_currentBatchIdx == 0) ? m_indices.size() - 1 : m_currentBatchIdx - 1;
        return m_cachedTargets[m_indices[lastIdx]];
    }

    /** @brief Signal that the first epoch is finished and the cache is ready. */
    void lockCache() {
        if (m_cachedInputs.empty()) return;
        m_isFull = true;
        m_indices.resize(m_cachedInputs.size());
        std::iota(m_indices.begin(), m_indices.end(), 0);
        
        // Initial shuffle
        if (m_shuffleEnabled) {
            std::shuffle(m_indices.begin(), m_indices.end(), m_rng);
        }
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        // Cache layer doesn't transform data, just passes gradients back
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

    bool isFull() const { return m_isFull; }
    void setFull(bool full) { m_isFull = full; }

private:
    bool m_shuffleEnabled;
    bool m_isFull;
    size_t m_inputDim = 0;
    
    std::vector<owTensor<float, 2>> m_cachedInputs;
    std::vector<owTensor<float, 2>> m_cachedTargets;
    std::vector<size_t> m_indices;
    size_t m_currentBatchIdx;

    std::mt19937 m_rng{std::random_device{}()};
};

} // namespace ow
