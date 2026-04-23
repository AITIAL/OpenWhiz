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
 * @brief Performance optimizer that records pre-processed data and replays it.
 * 
 * This layer is designed to be placed after non-trainable preprocessing layers 
 * (like owNormalizationLayer or owSlidingWindowLayer).
 * 
 * OPERATIONAL MODES:
 * 1. RECORDING (Epoch 1): Acts as a pass-through layer while storing every 
 *    input and target tensor in RAM.
 * 2. PLAYBACK (Epochs 2+): Ignores its input and returns stored tensors from memory.
 *    This dramatically speeds up training by avoiding repetitive preprocessing.
 * 3. PASS-THROUGH (Inference): After training, it returns its input directly,
 *    enabling real-time prediction on new data.
 */
class owCacheLayer : public owLayer {
public:
    owCacheLayer(bool shuffle = true) 
        : m_shuffleEnabled(shuffle), m_isFull(false), m_playbackMode(false), m_currentBatchIdx(0) {
        m_layerName = "Cache Layer";
    }

    size_t getInputSize() const override { return m_inputDim; }
    size_t getOutputSize() const override { return m_inputDim; }
    void setInputSize(size_t size) override { m_inputDim = size; }
    void setNeuronNum(size_t num) override { m_inputDim = num; }

    /**
     * @brief Resets the playback pointer to the start of the cache.
     * Shuffles indices if enabled to provide randomized training batches.
     */
    void reset() override {
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

    /**
     * @brief Core mode logic: Store in epoch 1, Replay in epoch 2+, Pass-through for inference.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        if (!m_isFull && m_isTraining) {
            // --- RECORDING MODE ---
            // Only record during the first training epoch
            m_cachedInputs.push_back(input);
            if (m_localTarget) m_cachedTargets.push_back(*m_localTarget); 
            m_inputDim = input.shape()[1];
            return input;
        } else if (m_playbackMode && m_isTraining) {
            // --- PLAYBACK MODE (High Speed Training) ---
            // Only used during training epochs 2-N
            if (m_indices.empty()) return input; 
            size_t idx = m_indices[m_currentBatchIdx];
            m_currentBatchIdx = (m_currentBatchIdx + 1) % m_indices.size();
            return m_cachedInputs[idx];
        } else {
            // --- PASS-THROUGH MODE (Inference / Evaluation / Validation) ---
            return input;
        }
    }

    /**
     * @brief Returns the target tensor corresponding to the currently replayed input.
     * This ensures the loss function compares the replayed input with the correct target.
     */
    const owTensor<float, 2>& getActiveTarget() const {
        if (!m_isFull || m_cachedTargets.empty()) {
            static owTensor<float, 2> empty;
            return empty;
        }
        size_t lastIdx = (m_currentBatchIdx == 0) ? m_indices.size() - 1 : m_currentBatchIdx - 1;
        return m_cachedTargets[m_indices[lastIdx]];
    }

    /** @brief Locks the buffer and initializes training indices. */
    void lockCache() override {
        if (m_cachedInputs.empty()) return;
        m_isFull = true;
        m_playbackMode = true; 
        m_indices.resize(m_cachedInputs.size());
        std::iota(m_indices.begin(), m_indices.end(), 0);
        if (m_shuffleEnabled) std::shuffle(m_indices.begin(), m_indices.end(), m_rng);
    }

    /** @brief Toggles between active playback and transparent mode. */
    void setPlaybackMode(bool enabled) override {
        m_playbackMode = enabled;
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        // Transparent pass-through for gradients
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
    bool m_playbackMode;
    size_t m_inputDim = 0;
    
    std::vector<owTensor<float, 2>> m_cachedInputs;
    std::vector<owTensor<float, 2>> m_cachedTargets;
    std::vector<size_t> m_indices;
    size_t m_currentBatchIdx;

    std::mt19937 m_rng{std::random_device{}()};
};

} // namespace ow
