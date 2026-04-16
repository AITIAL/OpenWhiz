/*
 * owConcatenateLayer.hpp
 *
 *  Created on: Apr 10, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
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
     * @brief Constructs an owConcatenateLayer with an optional initial set of branches.
     * @param branches A vector of shared pointers to the layers that will form the parallel branches.
     */
    owConcatenateLayer(const std::vector<std::shared_ptr<owLayer>>& branches = {}, bool useSharedInput = false)
        : m_branches(branches), m_useSharedInput(useSharedInput) {
        m_layerName = "Concatenate Layer";
    }

    /**
     * @brief Returns the list of internal branches.
     */
    std::vector<std::shared_ptr<owLayer>>& getBranches() { return m_branches; }

    /**
     * @brief Sets whether all branches should receive the full input (Shared) 
     * or a horizontal slice (Standard).
     * @param shared True for shared input, false for sliced input.
     */
    void setUseSharedInput(bool shared) { m_useSharedInput = shared; }

    /**
     * @brief Adds a new parallel branch to the layer.
     * @param branch Shared pointer to the layer to be added as a branch.
     */
    void addBranch(std::shared_ptr<owLayer> branch) {
        if (branch) m_branches.push_back(branch);
    }

    /**
     * @brief Replaces the current branches with a new set.
     * @details This is primarily used during XML deserialization to reconstruct complex architectures.
     * @param branches Vector of shared pointers to the new layers.
     */
    void setBranches(const std::vector<std::shared_ptr<owLayer>>& branches) {
        m_branches = branches;
    }

    /**
     * @brief Calculates the total input size expected by this layer.
     * @return The sum of the input sizes of all branches.
     */
    size_t getInputSize() const override {
        if (m_useSharedInput) {
            return m_branches.empty() ? 0 : m_branches[0]->getInputSize();
        }
        size_t total = 0;
        for (const auto& b : m_branches) total += b->getInputSize();
        return total;
    }

    /**
     * @brief Calculates the total output size produced by this layer.
     * @return The sum of the output sizes of all branches.
     */
    size_t getOutputSize() const override {
        size_t total = 0;
        for (const auto& b : m_branches) total += b->getOutputSize();
        return total;
    }

    /**
     * @brief Implementation of virtual setNeuronNum. 
     * @note Concatenate layer's "neurons" are determined by its branches; this call is ignored.
     */
    void setNeuronNum(size_t num) override { (void)num; }

    /**
     * @brief Performs the forward pass.
     * @param input Input tensor of shape [BatchSize, TotalInputSize].
     * @return Concatenated output tensor of shape [BatchSize, TotalOutputSize].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batch = input.shape()[0];
        m_outputs.clear();
        m_outputs.reserve(m_branches.size());

        size_t currentInOffset = 0;
        for (auto& branch : m_branches) {
            if (m_useSharedInput) {
                // Shared Input Mode: Pass full input to all branches
                m_outputs.push_back(branch->forward(input));
            } else {
                // Standard Slicing Mode: Split input features horizontally
                size_t inSize = branch->getInputSize();
                owTensor<float, 2> slicedInput(batch, inSize);
                for (size_t i = 0; i < batch; ++i) {
                    for (size_t j = 0; j < inSize; ++j) {
                        slicedInput(i, j) = input(i, currentInOffset + j);
                    }
                }
                m_outputs.push_back(branch->forward(slicedInput));
                currentInOffset += inSize;
            }
        }

        // Concatenate all branch outputs
        owTensor<float, 2> result(batch, getOutputSize());
        size_t currentOutOffset = 0;
        for (const auto& out : m_outputs) {
            size_t outSize = out.shape()[1];
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < outSize; ++j) {
                    result(i, currentOutOffset + j) = out(i, j);
                }
            }
            currentOutOffset += outSize;
        }

        return result;
    }

    /**
     * @brief Performs the backward pass.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batch = outputGradient.shape()[0];
        std::vector<owTensor<float, 2>> inputGradients;
        inputGradients.reserve(m_branches.size());

        size_t currentOutOffset = 0;
        for (size_t k = 0; k < m_branches.size(); ++k) {
            size_t outSize = m_outputs[k].shape()[1];
            
            // Slice output gradient
            owTensor<float, 2> slicedGrad(batch, outSize);
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < outSize; ++j) {
                    slicedGrad(i, j) = outputGradient(i, currentOutOffset + j);
                }
            }
            
            inputGradients.push_back(m_branches[k]->backward(slicedGrad));
            currentOutOffset += outSize;
        }

        if (m_useSharedInput) {
            // Shared Input Mode: Sum all branch gradients for the same input
            if (inputGradients.empty()) return owTensor<float, 2>(0, 0);
            
            size_t inSize = inputGradients[0].shape()[1];
            owTensor<float, 2> result(batch, inSize);
            result.setZero();

            // For shared input, we assume all branches receive the same size input.
            for (const auto& grad : inputGradients) {
                result = result + grad;
            }
            return result;
        } else {
            // Standard Slicing Mode: Concatenate input gradients
            owTensor<float, 2> result(batch, getInputSize());
            size_t currentInOffset = 0;
            for (const auto& inGrad : inputGradients) {
                size_t inSize = inGrad.shape()[1];
                for (size_t i = 0; i < batch; ++i) {
                    for (size_t j = 0; j < inSize; ++j) {
                        result(i, currentInOffset + j) = inGrad(i, j);
                    }
                }
                currentInOffset += inSize;
            }
            return result;
        }
    }

    /**
     * @brief Triggers the training update for internal branches, respecting freezing state.
     */
    void train() override {
        if (m_isFrozen) return;
        for (auto& branch : m_branches) {
            if (!branch->isFrozen()) branch->train();
        }
    }

    /**
     * @brief Assigns an optimizer to this layer and all its branches.
     * @param opt Pointer to the optimizer.
     */
    void setOptimizer(owOptimizer* opt) override {
        owLayer::setOptimizer(opt);
        for (auto& branch : m_branches) {
            branch->setOptimizer(opt);
        }
    }

    /**
     * @brief Resets all internal branches.
     */
    void reset() override {
        for (auto& branch : m_branches) branch->reset();
    }

    /**
     * @brief Creates a deep copy of the concatenate layer and all its branches.

     * @return Shared pointer to the cloned layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        std::vector<std::shared_ptr<owLayer>> branchCopies;
        for (const auto& b : m_branches) branchCopies.push_back(b->clone());
        auto copy = std::make_shared<owConcatenateLayer>(branchCopies, m_useSharedInput);
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes the layer and its recursive branch structure to XML.
     * @return XML string representing the layer.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<BranchCount>" << m_branches.size() << "</BranchCount>\n";
        ss << "<UseSharedInput>" << (m_useSharedInput ? 1 : 0) << "</UseSharedInput>\n";
        for (size_t i = 0; i < m_branches.size(); ++i) {
            ss << "<Branch_" << i << " type=\"" << m_branches[i]->getLayerName() << "\">\n" 
               << m_branches[i]->toXML() << "</Branch_" << i << ">\n";
        }
        return ss.str();
    }

    /**
     * @brief Deserializes the branch parameters from XML.
     * @note Structural reconstruction is handled by owNeuralNetwork::loadFromXML.
     * @param xml XML content for the layer.
     */
    void fromXML(const std::string& xml) override {
        std::string sharedVal = owLayer::getTagContent(xml, "UseSharedInput");
        if (!sharedVal.empty()) m_useSharedInput = (std::stoi(sharedVal) == 1);
        
        for (size_t i = 0; i < m_branches.size(); ++i) {
            std::string tag = "Branch_" + std::to_string(i);
            m_branches[i]->fromXML(owLayer::getNestedTagContent(xml, tag));
        }
    }

    /** @return Always nullptr as this is a meta-layer. */
    float* getParamsPtr() override { return nullptr; }
    /** @return Always nullptr as this is a meta-layer. */
    float* getGradsPtr() override { return nullptr; }
    /** @return Always 0 as this layer only delegates to its branches. */
    size_t getParamsCount() override { return 0; }

private:
    std::vector<std::shared_ptr<owLayer>> m_branches; ///< Internal storage for parallel branches.
    std::vector<owTensor<float, 2>> m_outputs;      ///< Cached outputs of branches for the backward pass.
    bool m_useSharedInput = false; ///< Flag for Multi-Scale View mode.
};

} // namespace ow
