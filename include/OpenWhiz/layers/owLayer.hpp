/*
 * owLayer.hpp
 *
 *  Created on: Nov 24, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "../core/owTensor.hpp"
#include "../core/owCuda.hpp"
#include <memory>
#include <string>
#include <sstream>
#include "../nonlinearities/owActivation.hpp"
#include "../nonlinearities/owIdentityActivation.hpp"
#include "../nonlinearities/owReLUActivation.hpp"
#include "../nonlinearities/owSigmoidActivation.hpp"
#include "../nonlinearities/owTanhActivation.hpp"
#include "../nonlinearities/owLeakyReLUActivation.hpp"

#ifndef OW_PI
#define OW_PI 3.14159265358979323846f
#endif


namespace ow {

// Forward declaration to break circular dependency
class owOptimizer;
class owNeuralNetwork;

enum owRegularizationType {
    NONE = 0,
    L1 = 1,
    L2 = 2
};

/**
 * @class owLayer
 * @brief Abstract base class for all neural network layers in OpenWhiz.
 * 
 * A layer is the fundamental block of a neural network. It performs 
 * a transformation on the input data (forward pass) and computes gradients 
 * (backward pass) for optimization.
 * 
 * @details
 * Each layer in OpenWhiz handles:
 * - Forward propagation: Transforming input tensors to output tensors.
 * - Backward propagation: Computing gradients with respect to inputs and parameters.
 * - Parameter management: Storing and updating weights and biases.
 * - Serialization: Converting layer state to and from XML.
 * 
 * @note
 * Layers are designed to be composable. When building a network, ensure that 
 * the output size of one layer matches the input size of the next.
 */
class owLayer {
public:
    virtual ~owLayer() = default;

    /**
     * @brief Performs the forward pass.
     * @param input The input tensor (typically [BatchSize, InputSize]).
     * @return The transformed output tensor (typically [BatchSize, OutputSize]).
     */
    virtual owTensor<float, 2> forward(const owTensor<float, 2>& input) = 0;

    /**
     * @brief Performs the backward pass to compute gradients.
     * @param outputGradient The gradient of the loss with respect to the output of this layer.
     * @return The gradient of the loss with respect to the input of this layer.
     */
    virtual owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) = 0;

    /**
     * @brief Updates layer parameters using the assigned optimizer.
     * 
     * This method is typically called after the backward pass during the 
     * training phase.
     */
    virtual void train() = 0;

    /** @return The number of input features expected by this layer. */
    virtual size_t getInputSize() const = 0;

    /** @brief Sets the input feature size. This may trigger internal parameter reinitialization. */
    virtual void setInputSize(size_t size) { (void)size; }

    /** @return The number of output features produced by this layer. */
    virtual size_t getOutputSize() const = 0;

    /** @return The number of neurons in the layer (defaults to output size). */
    virtual size_t getNeuronNum() const { return getOutputSize(); }

    /** @brief Sets the number of neurons in the layer. */
    virtual void setNeuronNum(size_t num) = 0;

    /** @return A shared pointer to a deep copy of this layer. */
    virtual std::shared_ptr<owLayer> clone() const = 0;

    /** @return The name of the layer type. */
    std::string getLayerName() const { return m_layerName; }

    /** @brief Sets the name of the layer. */
    void setLayerName(const std::string& name) { m_layerName = name; }

    /** @brief Resets any internal state (useful for recurrent or stateful layers). */
    virtual void reset() {}

    /** @return XML representation of the layer's configuration and parameters. */
    virtual std::string toXML() const = 0;

    /** @brief Reconstructs the layer state from an XML string. */
    virtual void fromXML(const std::string& xml) = 0;

    /** @brief Static helper to extract content between XML tags. */
    static std::string getTagContent(const std::string& s, const std::string& tag) {
        std::string sT = "<" + tag; 
        size_t s_pos = s.find(sT);
        if (s_pos == std::string::npos) return std::string("");
        size_t close_bracket = s.find(">", s_pos);
        size_t e_pos = s.find("</" + tag + ">", close_bracket);
        if (e_pos == std::string::npos) return std::string("");
        return s.substr(close_bracket + 1, e_pos - (close_bracket + 1));
    }

    /** @brief Static helper to extract content of a tag including its attributes. */
    static std::string getTagContentWithAttributes(const std::string& s, const std::string& tag) {
        std::string sT = "<" + tag; 
        size_t s_pos = s.find(sT);
        if (s_pos == std::string::npos) return std::string("");
        size_t e_tag_pos = s.find("</" + tag + ">", s_pos);
        if (e_tag_pos == std::string::npos) return std::string("");
        return s.substr(s_pos, (e_tag_pos + tag.length() + 3) - s_pos);
    }

    /** @brief Static helper to extract an attribute value from an XML tag. */
    static std::string getAttr(const std::string& s, const std::string& attr) {
        std::string aS = attr + "=\"";
        size_t pos = s.find(aS);
        if (pos == std::string::npos) return std::string("");
        size_t end = s.find("\"", pos + aS.length());
        return s.substr(pos + aS.length(), end - (pos + aS.length()));
    }

    /** @brief Static helper to extract content of a tag that might be nested. */
    static std::string getNestedTagContent(const std::string& s, const std::string& tag) {
        std::string openTag = "<" + tag;
        std::string closeTag = "</" + tag + ">";
        
        size_t startPos = s.find(openTag);
        if (startPos == std::string::npos) return "";
        
        size_t contentStart = s.find(">", startPos);
        if (contentStart == std::string::npos) return "";
        contentStart++;
        
        int depth = 1;
        size_t currentPos = contentStart;
        while (depth > 0) {
            size_t nextOpen = s.find(openTag, currentPos);
            size_t nextClose = s.find(closeTag, currentPos);
            
            if (nextClose == std::string::npos) return ""; // Malformed
            
            if (nextOpen != std::string::npos && nextOpen < nextClose) {
                // Check if it's a real open tag (not a substring)
                char nextChar = s[nextOpen + openTag.length()];
                if (nextChar == '>' || nextChar == ' ') {
                    depth++;
                    currentPos = nextOpen + openTag.length();
                    continue;
                }
            }
            
            depth--;
            if (depth == 0) return s.substr(contentStart, nextClose - contentStart);
            currentPos = nextClose + closeTag.length();
        }
        return "";
    }
    
    /** @brief Assigns an optimizer to this layer for parameter updates. */
    virtual void setOptimizer(owOptimizer* opt) { m_optimizer = opt; }

    /** @return The optimizer currently assigned to this layer. */
    owOptimizer* getOptimizer() { return m_optimizer; }
    
    /** @brief Sets the activation function for this layer. */
    void setActivation(std::shared_ptr<owActivation> act) { m_activation = act; }

    /** @return The activation function currently used by this layer. */
    owActivation* getActivation() { return m_activation.get(); }
    
    /** @brief Sets the regularization type (None, L1, L2). */
    virtual void setRegularization(int type) { m_regType = static_cast<owRegularizationType>(type); }

    /** @return The current regularization type. */
    int getRegularization() const { return static_cast<int>(m_regType); }

    /** @brief Sets the regularization strength (lambda). */
    void setRegularizationLambda(float lambda) { m_regLambda = lambda; }

    /** @return The string name of the current activation function. */
    std::string getActivationName() const {
        if (std::dynamic_pointer_cast<owReLUActivation>(m_activation)) return "ReLU";
        if (std::dynamic_pointer_cast<owSigmoidActivation>(m_activation)) return "Sigmoid";
        if (std::dynamic_pointer_cast<owTanhActivation>(m_activation)) return "Tanh";
        if (std::dynamic_pointer_cast<owLeakyReLUActivation>(m_activation)) return "LeakyReLU";
        return "Identity";
    }

    /** @brief Sets the activation function by its string name. */
    void setActivationByName(const std::string& name) {
        if (name == "ReLU") m_activation = std::make_shared<owReLUActivation>();
        else if (name == "Sigmoid") m_activation = std::make_shared<owSigmoidActivation>();
        else if (name == "Tanh") m_activation = std::make_shared<owTanhActivation>();
        else if (name == "LeakyReLU") m_activation = std::make_shared<owLeakyReLUActivation>();
        else m_activation = std::make_shared<owIdentityActivation>();
    }

    /** @brief Prevents this layer from updating its parameters. */
    void setFrozen(bool frozen) { m_isFrozen = frozen; }

    /** @return True if the layer's learning is currently disabled. */
    bool isFrozen() const { return m_isFrozen; }

    /** @brief Toggles whether the layer is active or acts as a pass-through identity. */
    void setEnabled(bool enabled) { m_enabled = enabled; }

    /** @return True if the layer is currently active. */
    bool isEnabled() const { return m_enabled; }

    /** @brief Enables or disables independent expert training for this layer. */
    void setIndependentExpertMode(bool enable) { m_isIndependentExpertMode = enable; }

    /** @return True if this layer is acting as an independent expert. */
    bool isIndependentExpertMode() const { return m_isIndependentExpertMode; }

    /** @brief Sets the weight of the local expertise gradient (0.0 to 1.0+). */
    void setLocalExpertWeight(float weight) { m_localExpertWeight = weight; }

    /** @return Current local expertise weight. */
    float getLocalExpertWeight() const { return m_localExpertWeight; }

    /** @brief Sets a local convergence threshold. If local error is below this, the layer can auto-freeze. */
    void setConvergenceThreshold(float threshold) { m_convergenceThreshold = threshold; }
    
    /** @return Current convergence threshold. */
    float getConvergenceThreshold() const { return m_convergenceThreshold; }

    /** @brief Sets the target tensor pointer for independent supervised training of this layer/branch. */
    virtual void setTarget(const owTensor<float, 2>* target) { m_localTarget = target; }

    /** @return Pointer to the raw parameter data (for global optimization). */
	virtual float* getParamsPtr() = 0;

    /** @return Pointer to the raw gradient data. */
	virtual float* getGradsPtr() = 0;

    /** @return Total number of trainable parameters in this layer. */
	virtual size_t getParamsCount() = 0;

    /** @brief Synchronizes layer state (useful for distributed training). */
	virtual void synchronize() {};

    /** @brief Sets the parent neural network that contains this layer. */
    void setParentNetwork(owNeuralNetwork* nn) { m_parentNetwork = nn; }

    /** @return Pointer to the parent neural network. */
    owNeuralNetwork* getParentNetwork() const { return m_parentNetwork; }

    /** @brief Toggles training mode for this layer. */
    virtual void setTraining(bool training) { m_isTraining = training; }

    /** @return True if the layer is currently in training mode. */
    bool isTraining() const { return m_isTraining; }

    /** @brief Transitions the layer from recording to playback (for caching). */
    virtual void lockCache() {}

    /** @brief Toggles playback mode for caching layers. */
    virtual void setPlaybackMode(bool /*enabled*/) {}

    /** @return True if the cache is fully populated. */
    virtual bool isFull() const { return false; }

    /** @return The current target tensor associated with the cached data. */
    virtual const owTensor<float, 2>& getActiveTarget() const {
        static owTensor<float, 2> empty;
        return empty;
    }

protected:
    std::string m_layerName = "Base Layer";
    owNeuralNetwork* m_parentNetwork = nullptr;
    bool m_isIndependentExpertMode = false;
    bool m_isFrozen = false; 
    bool m_isTraining = true; 
    bool m_enabled = true; 
    float m_convergenceThreshold = 0.0f;
    const owTensor<float, 2>* m_localTarget = nullptr; 
    float m_localExpertWeight = 1.0f;
    owOptimizer* m_optimizer = nullptr;
    std::shared_ptr<owActivation> m_activation = std::make_shared<owIdentityActivation>();
    owRegularizationType m_regType = L2;
    float m_regLambda = 0.01f;

    void applyRegularization(owTensor<float, 2>& weights, owTensor<float, 2>& gradients) {
        if (m_regType == NONE || m_regLambda == 0) return;

    #ifdef OW_USE_GPU
        cuda::applyRegularization(weights.data(), gradients.data(), (int)weights.size(), static_cast<int>(m_regType), m_regLambda);
    #else
        for (size_t i = 0; i < weights.size(); ++i) {
            if (m_regType == L2) gradients.data()[i] += m_regLambda * weights.data()[i];
            else if (m_regType == L1) gradients.data()[i] += m_regLambda * (weights.data()[i] > 0 ? 1.0f : -1.0f);
        }
    #endif
    }
};

} // namespace ow
