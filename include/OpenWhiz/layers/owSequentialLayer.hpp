/*
 * owSequentialLayer.hpp
 *
 *  Created on: Apr 12, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <vector>
#include <memory>

namespace ow {

/**
 * @class owSequentialLayer
 * @brief A container layer that executes a sequence of layers internally.
 * 
 * The Sequential Layer manages a list of sub-layers and processes data through them
 * in the order they were added. It also supports "Independent Expert Mode", 
 * where a local target can be used to inject gradients directly into the sequence,
 * allowing sub-components to learn specialized features (e.g., in hierarchical or 
 * multi-task learning setups).
 */
class owSequentialLayer : public owLayer {
public:
    /**
     * @brief Constructs an empty owSequentialLayer.
     */
    owSequentialLayer() {
        m_layerName = "Sequential Layer";
    }

    /**
     * @brief Adds a layer to the end of the sequence.
     * @param layer A shared pointer to the layer to be added.
     */
    void addLayer(std::shared_ptr<owLayer> layer) {
        if (layer) m_layers.push_back(layer);
    }

    /**
     * @brief Returns the input size of the first layer in the sequence.
     * @return size_t Input size, or 0 if empty.
     */
    size_t getInputSize() const override {
        return m_layers.empty() ? 0 : m_layers.front()->getInputSize();
    }

    /**
     * @brief Returns the output size of the last layer in the sequence.
     * @return size_t Output size, or 0 if empty.
     */
    size_t getOutputSize() const override {
        return m_layers.empty() ? 0 : m_layers.back()->getOutputSize();
    }

    /**
     * @brief Sets the number of neurons for the first layer in the sequence.
     * @param num Number of neurons.
     */
    void setNeuronNum(size_t num) override {
        if (!m_layers.empty()) m_layers.front()->setNeuronNum(num);
    }

    /**
     * @brief Forward pass: sequentially executes all internal layers.
     * 
     * Formula: @f$ y = f_n(\dots f_2(f_1(x))\dots) @f$
     * 
     * @param input Input tensor.
     * @return owTensor<float, 2> Output of the last layer.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> output = input;
        for (auto& layer : m_layers) output = layer->forward(output);
        m_lastOutput = output; // Cache for local loss
        return output;
    }

    /**
     * @brief Backward pass: propagates gradients in reverse order and optionally injects local expertise.
     * 
     * If "Independent Expert Mode" is active, a local gradient based on the local target is added:
     * 1. Local Loss (MSE): @f$ L_{local} = \frac{1}{2N} \sum (y - \text{target})^2 @f$
     * 2. Local Gradient: @f$ \frac{\partial L_{local}}{\partial y} = \frac{y - \text{target}}{N} \times \text{weight} @f$
     * 3. Combined Gradient: @f$ G_{total} = G_{external} + G_{local} @f$
     * 
     * This @f$ G_{total} @f$ is then propagated backward through the internal layers.
     * 
     * @param outputGradient Gradient of loss w.r.t. the output of the sequence.
     * @return owTensor<float, 2> Gradient w.r.t. the input of the sequence.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> grad = outputGradient;
        
        // --- Independent Expertise Injection ---
        // Only inject local gradients if this layer is in Independent Expert Mode!
        if (m_isIndependentExpertMode && m_localTarget && !m_isFrozen && m_lastOutput.size() > 0) {
            size_t batchSize = m_lastOutput.shape()[0];
            
            // Assuming MSE loss for local expertise: L = 0.5 * (out - target)^2
            // dL/dout = (out - target) * weight
            if (m_localTarget->size() == m_lastOutput.size()) {
                for (size_t i = 0; i < grad.size(); ++i) {
                    float localGrad = (m_lastOutput.data()[i] - m_localTarget->data()[i]) / (float)batchSize;
                    grad.data()[i] += localGrad * m_localExpertWeight;
                }
            }
        }

        // Propagate combined gradient through internal layers
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }

    /**
     * @brief Computes Mean Squared Error (MSE) between the internal output and a local target.
     * 
     * Formula: @f$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 @f$
     * 
     * @param target The local target tensor to compare against.
     * @return float Calculated MSE value.
     */
    float computeLocalLoss(const owTensor<float, 2>& target) {
        if (m_lastOutput.size() == 0 || target.size() == 0 || target.size() != m_lastOutput.size()) return 1e30f;
        float mse = 0;
        size_t n = m_lastOutput.size();
        for (size_t i = 0; i < n; ++i) {
            float diff = m_lastOutput.data()[i] - target.data()[i];
            mse += diff * diff;
        }
        return mse / (float)n;
    }

    /**
     * @brief Manually triggers a training step using only the local target.
     * 
     * This method computes the local gradient @f$ (y - \text{target}) / N @f$, 
     * performs a backward pass through the sequence, and then calls `train()` 
     * to update internal parameters. It is used for specialized pre-training or 
     * auxiliary learning.
     */
    void trainIndependentExpertOnly() {
        if (!m_isIndependentExpertMode || !m_localTarget || m_isFrozen || m_lastOutput.size() == 0) return;
        
        size_t batchSize = m_lastOutput.shape()[0];
        owTensor<float, 2> grad(m_lastOutput.shape()[0], m_lastOutput.shape()[1]);
        
        for (size_t i = 0; i < grad.size(); ++i) {
            grad.data()[i] = (m_lastOutput.data()[i] - m_localTarget->data()[i]) / (float)batchSize;
        }

        backward(grad);
        train();
    }

    /**
     * @brief Updates all internal layers using their respective optimizers.
     */
    void train() override {
        if (m_isFrozen) return; // Independent stopping!
        for (auto& layer : m_layers) layer->train();
    }

    /**
     * @brief Resets all internal layers.
     */
    void reset() override {
        for (auto& layer : m_layers) layer->reset();
    }

    /**
     * @brief Sets the optimizer for the sequential layer and all its sub-layers.
     * @param opt Pointer to the optimizer.
     */
    void setOptimizer(owOptimizer* opt) override {
        owLayer::setOptimizer(opt);
        for (auto& layer : m_layers) layer->setOptimizer(opt);
    }

    /**
     * @brief Sets the regularization type for the sequential layer and all sub-layers.
     * @param type Regularization type constant.
     */
    void setRegularization(int type) override {
        owLayer::setRegularization(type);
        for (auto& layer : m_layers) layer->setRegularization(type);
    }

    /**
     * @brief Returns nullptr as Sequential Layer doesn't hold parameters directly.
     * @return float* Always nullptr.
     */
    float* getParamsPtr() override { return nullptr; } 

    /**
     * @brief Returns nullptr as Sequential Layer doesn't hold gradients directly.
     * @return float* Always nullptr.
     */
    float* getGradsPtr() override { return nullptr; }

    /**
     * @brief Returns the total count of parameters across all sub-layers.
     * @return size_t Total parameter count.
     */
    size_t getParamsCount() override {
        size_t total = 0;
        for (const auto& l : m_layers) total += l->getParamsCount();
        return total;
    }

    /**
     * @brief Creates a deep copy of the Sequential Layer and its sub-layers.
     * @return std::shared_ptr<owLayer> Cloned layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owSequentialLayer>();
        for (const auto& l : m_layers) copy->addLayer(l->clone());
        copy->m_isFrozen = m_isFrozen;
        copy->m_convergenceThreshold = m_convergenceThreshold;
        copy->m_isIndependentExpertMode = m_isIndependentExpertMode;
        return copy;
    }

    /**
     * @brief Serializes the layer and all its sub-layers to XML.
     * @return std::string XML representation.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<LayerCount>" << m_layers.size() << "</LayerCount>\n";
        ss << "<Frozen>" << (m_isFrozen ? 1 : 0) << "</Frozen>\n";
        ss << "<IndependentExpertMode>" << (m_isIndependentExpertMode ? 1 : 0) << "</IndependentExpertMode>\n";
        ss << "<Threshold>" << m_convergenceThreshold << "</Threshold>\n";
        for (size_t i = 0; i < m_layers.size(); ++i) {
            ss << "<SubLayer_" << i << " type=\"" << m_layers[i]->getLayerName() << "\">\n"
               << m_layers[i]->toXML() << "</SubLayer_" << i << ">\n";
        }
        return ss.str();
    }

    /**
     * @brief Deserializes the layer state and propagates XML to sub-layers.
     * @param xml XML string.
     */
    void fromXML(const std::string& xml) override {
        m_isFrozen = (getTagContent(xml, "Frozen") == "1");
        m_isIndependentExpertMode = (getTagContent(xml, "IndependentExpertMode") == "1");
        std::string thr = getTagContent(xml, "Threshold");
        if (!thr.empty()) m_convergenceThreshold = std::stof(thr);
        for (size_t i = 0; i < m_layers.size(); ++i) {
            std::string tag = "SubLayer_" + std::to_string(i);
            m_layers[i]->fromXML(owLayer::getNestedTagContent(xml, tag));
        }
    }

private:
    std::vector<std::shared_ptr<owLayer>> m_layers;
    owTensor<float, 2> m_lastOutput;
};

} // namespace ow
