/*
 * owConcatenateLayer.hpp
 *
 *  Created on: Apr 10, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include "owSequentialLayer.hpp"
#include <vector>
#include <memory>
#include <numeric>

namespace ow {

/**
 * @class owConcatenateLayer
 * @brief A structural meta-layer that implements the "Independent Experts" architecture by parallelizing network branches.
 * 
 * This layer acts as a powerful orchestration point, allowing multiple independent neural network 
 * pathways (Experts) to process the same or sliced input data. The results are merged into 
 * a single wide feature vector for final decision-making.
 * 
 * **Architectural Significance:**
 * - **Multi-Scale Analysis:** Allows implementing branches that look at different temporal 
 *   resolutions (e.g., Weekly vs. Monthly trends in Financial Systems).
 * - **Modular Design:** Experts can be complex sequences of layers, including LSTMs or 
 *   PCA-enhanced blocks.
 * 
 * **Mathematical Logic:**
 * - **Forward Pass:** 
 *   Let $X$ be the input. The output is a horizontal concatenation:
 *   $Y = [Expert_1(X), Expert_2(X), ..., Expert_N(X)]$
 *   Total Output Width = $\sum OutputSize_i$.
 * - **Backward Pass (Gradient Flow):**
 *   The upstream gradient $\delta$ is sliced into $\{\delta_1, \delta_2, ..., \delta_N\}$. 
 *   If `m_useSharedInput` is enabled, the resulting input gradient is the element-wise sum:
 *   $\nabla X = \sum_{i=1}^{N} \nabla Expert_i(\delta_i)$
 */
class owConcatenateLayer : public owLayer {
public:
    /**
     * @class owBranch
     * @brief Encapsulates a parallel network pathway (Expert) as a sequential layer sequence.
     * 
     * In OpenWhiz, every expert branch is architecturaly a sequence of layers.
     * owBranch inherits from owSequentialLayer to provide a unified interface for 
     * building these expert pipelines.
     */
    class owBranch : public owSequentialLayer {
    public:
        owBranch() : owSequentialLayer() {
            m_layerName = "Branch";
        }

        /**
         * @brief Enables or disables the expert branch.
         */
        void setEnabled(bool enable) { m_enabled = enable; }

        /**
         * @brief Checks if the branch is currently active.
         */
        bool isEnabled() const { return m_enabled; }

    private:
        bool m_enabled = true;            ///< Active status of the expert.
    };

    /**
     * @brief Constructs an owConcatenateLayer with optional initial expert branches.
     */
    owConcatenateLayer(const std::vector<std::shared_ptr<owBranch>>& branches = {}, bool useSharedInput = false)
        : m_branches(branches), m_useSharedInput(useSharedInput) {
        m_layerName = "Concatenate Layer";
    }

    /**
     * @brief Returns the list of internal branch objects.
     */
    std::vector<std::shared_ptr<owBranch>>& getBranches() { return m_branches; }

    /**
     * @brief Returns a specific branch by its index.
     * @param branchNo Index of the branch.
     * @return Shared pointer to the branch, or nullptr if index is invalid.
     */
    std::shared_ptr<owBranch> getBranch(int branchNo) {
        if (branchNo >= 0 && (size_t)branchNo < m_branches.size()) {
            return m_branches[branchNo];
        }
        return nullptr;
    }

    /**
     * @brief Replaces a specific branch with a new expert.
     * @param branchNo Index of the branch to replace.
     * @param branch Shared pointer to the new branch.
     */
    void setBranch(int branchNo, std::shared_ptr<owBranch> branch) {
        if (branchNo >= 0 && (size_t)branchNo < m_branches.size() && branch) {
            m_branches[branchNo] = branch;
        }
    }

    /**
     * @brief Enables or disables a specific branch.
     */
    void enableBranch(int branchNo, bool enable) {
        if (branchNo >= 0 && (size_t)branchNo < m_branches.size()) {
            m_branches[branchNo]->setEnabled(enable);
        }
    }

    /**
     * @brief Checks if a specific branch is enabled.
     */
    bool isBranchEnabled(int branchNo) const {
        if (branchNo >= 0 && (size_t)branchNo < m_branches.size()) {
            return m_branches[branchNo]->isEnabled();
        }
        return false;
    }

    /**
     * @brief Sets whether all branches should receive the full input (Shared) 
     * or a horizontal slice (Standard).
     */
    void setUseSharedInput(bool shared) { m_useSharedInput = shared; }

    /**
     * @brief Adds an existing branch to the layer.
     */
    void addBranch(std::shared_ptr<owBranch> branch) {
        if (branch) m_branches.push_back(branch);
    }

    /**
     * @brief Automatically creates and adds a new expert branch.
     * @return A shared pointer to the newly created owBranch for configuration.
     */
    std::shared_ptr<owBranch> addBranch() {
        auto branch = std::make_shared<owConcatenateLayer::owBranch>();
        m_branches.push_back(branch);
        return branch;
    }

    /**
     * @brief Replaces the current branches with a new set of experts.
     */
    void setBranches(const std::vector<std::shared_ptr<owBranch>>& branches) {
        m_branches = branches;
    }

    /**
     * @brief Calculates total input size considering only enabled branches.
     */
    size_t getInputSize() const override {
        if (m_useSharedInput) {
            for (const auto& b : m_branches) if (b->isEnabled()) return b->getInputSize();
            return 0;
        }
        size_t total = 0;
        for (const auto& b : m_branches) if (b->isEnabled()) total += b->getInputSize();
        return total;
    }

    /**
     * @brief Calculates total output size considering only enabled branches.
     */
    size_t getOutputSize() const override {
        size_t total = 0;
        for (const auto& b : m_branches) if (b->isEnabled()) total += b->getOutputSize();
        return total;
    }

    void setNeuronNum(size_t num) override { (void)num; }

    /**
     * @brief Performs forward pass skipping disabled branches.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batch = input.shape()[0];
        m_activeOutputs.clear();
        m_activeBranchIndices.clear();

        size_t currentInOffset = 0;
        for (size_t i = 0; i < m_branches.size(); ++i) {
            if (!m_branches[i]->isEnabled()) {
                if (!m_useSharedInput) currentInOffset += m_branches[i]->getInputSize();
                continue;
            }

            m_activeBranchIndices.push_back(i);
            if (m_useSharedInput) {
                m_activeOutputs.push_back(m_branches[i]->forward(input));
            } else {
                size_t inSize = m_branches[i]->getInputSize();
                owTensor<float, 2> slicedInput(batch, inSize);
                for (size_t r = 0; r < batch; ++r) {
                    for (size_t c = 0; c < inSize; ++c) slicedInput(r, c) = input(r, currentInOffset + c);
                }
                m_activeOutputs.push_back(m_branches[i]->forward(slicedInput));
                currentInOffset += inSize;
            }
        }

        owTensor<float, 2> result(batch, getOutputSize());
        size_t currentOutOffset = 0;
        for (const auto& out : m_activeOutputs) {
            size_t outSize = out.shape()[1];
            for (size_t r = 0; r < batch; ++r) {
                for (size_t c = 0; c < outSize; ++c) result(r, currentOutOffset + c) = out(r, c);
            }
            currentOutOffset += outSize;
        }
        return result;
    }

    /**
     * @brief Performs backward pass routing gradients to active experts.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batch = outputGradient.shape()[0];
        std::vector<owTensor<float, 2>> activeInputGradients;

        size_t currentOutOffset = 0;
        for (size_t k = 0; k < m_activeOutputs.size(); ++k) {
            size_t outSize = m_activeOutputs[k].shape()[1];
            owTensor<float, 2> slicedGrad(batch, outSize);
            for (size_t r = 0; r < batch; ++r) {
                for (size_t c = 0; c < outSize; ++c) slicedGrad(r, c) = outputGradient(r, currentOutOffset + c);
            }
            
            size_t originalIdx = m_activeBranchIndices[k];
            activeInputGradients.push_back(m_branches[originalIdx]->backward(slicedGrad));
            currentOutOffset += outSize;
        }

        if (m_useSharedInput) {
            if (activeInputGradients.empty()) return owTensor<float, 2>(batch, getInputSize());
            size_t inSize = activeInputGradients[0].shape()[1];
            owTensor<float, 2> result(batch, inSize); result.setZero();
            for (const auto& grad : activeInputGradients) result = result + grad;
            return result;
        } else {
            owTensor<float, 2> result(batch, getInputSize());
            size_t currentInOffset = 0;
            for (const auto& inGrad : activeInputGradients) {
                size_t inSize = inGrad.shape()[1];
                for (size_t r = 0; r < batch; ++r) {
                    for (size_t c = 0; c < inSize; ++c) result(r, currentInOffset + c) = inGrad(r, c);
                }
                currentInOffset += inSize;
            }
            return result;
        }
    }

    void train() override {
        if (m_isFrozen) return;
        for (auto& b : m_branches) if (b->isEnabled() && !b->isFrozen()) b->train();
    }

    void setOptimizer(owOptimizer* opt) override {
        owLayer::setOptimizer(opt);
        for (auto& b : m_branches) b->setOptimizer(opt);
    }

    void reset() override {
        for (auto& b : m_branches) if (b->isEnabled()) b->reset();
    }

    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owConcatenateLayer>(std::vector<std::shared_ptr<owBranch>>(), m_useSharedInput);
        copy->m_layerName = m_layerName;
        for (const auto& b : m_branches) {
            auto bCopy = std::make_shared<owConcatenateLayer::owBranch>();
            bCopy->setEnabled(b->isEnabled());
            bCopy->setLayerName(b->getLayerName());
            bCopy->setFrozen(b->isFrozen());
            bCopy->setIndependentExpertMode(b->isIndependentExpertMode());
            bCopy->setConvergenceThreshold(b->getConvergenceThreshold());
            bCopy->fromXML(b->toXML()); 

            copy->m_branches.push_back(bCopy);
        }
        return copy;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<BranchCount>" << m_branches.size() << "</BranchCount>\n";
        ss << "<UseSharedInput>" << (m_useSharedInput ? 1 : 0) << "</UseSharedInput>\n";
        for (size_t i = 0; i < m_branches.size(); ++i) {
            ss << "<Branch_" << i << " type=\"" << m_branches[i]->getLayerName() 
               << "\" enabled=\"" << (m_branches[i]->isEnabled() ? 1 : 0) << "\">\n" 
               << m_branches[i]->toXML() << "</Branch_" << i << ">\n";
        }
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        std::string sharedVal = owLayer::getTagContent(xml, "UseSharedInput");
        if (!sharedVal.empty()) m_useSharedInput = (std::stoi(sharedVal) == 1);
        
        std::string countStr = owLayer::getTagContent(xml, "BranchCount");
        size_t count = countStr.empty() ? 0 : std::stoul(countStr);

        // Ensure we have enough branches
        while (m_branches.size() < count) m_branches.push_back(std::make_shared<owConcatenateLayer::owBranch>());
        
        for (size_t i = 0; i < count; ++i) {
            std::string tag = "Branch_" + std::to_string(i);
            std::string fullTag = owLayer::getTagContentWithAttributes(xml, tag);
            std::string enabledStr = owLayer::getAttr(fullTag, "enabled");
            if (!enabledStr.empty()) m_branches[i]->setEnabled(std::stoi(enabledStr) == 1);
            m_branches[i]->fromXML(owLayer::getNestedTagContent(xml, tag));
        }
    }

    float* getParamsPtr() override { return nullptr; }
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override { return 0; }

private:
    std::vector<std::shared_ptr<owBranch>> m_branches; ///< Encapsulated expert pathways.
    std::vector<owTensor<float, 2>> m_activeOutputs; ///< Cached outputs for backprop.
    std::vector<size_t> m_activeBranchIndices;       ///< Bookkeeping for enabled branches.
    bool m_useSharedInput = false;
};

} // namespace ow
