#pragma once

#include <vector>

#include "OpenWhiz/openwhiz.hpp"

namespace ow {

// Class-weighted categorical cross-entropy: same eps-clamped formula as
// owCategoricalCrossEntropyLoss, scaled per-example by its true class's
// weight. eps must match owProbabilityLayer's output clamp.
class owWeightedCategoricalCrossEntropyLoss : public owLoss {
public:
    explicit owWeightedCategoricalCrossEntropyLoss(std::vector<float> classWeights)
        : m_classWeights(std::move(classWeights)) {}

    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t batchSize = prediction.shape()[0];
        size_t numClasses = prediction.shape()[1];
        const float eps = 1e-12f;
        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t c = 0; c < numClasses; ++c) {
                if (target(b, c) > 0.5f) {
                    float p = std::max(eps, std::min(1.0f - eps, prediction(b, c)));
                    loss -= classWeightOf(c) * std::log(p);
                }
            }
        }
        return loss / static_cast<float>(batchSize);
    }

    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t batchSize = prediction.shape()[0];
        size_t numClasses = prediction.shape()[1];
        const float eps = 1e-12f;
        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t c = 0; c < numClasses; ++c) {
                float p = std::max(eps, std::min(1.0f - eps, prediction(b, c)));
                grad(b, c) = -classWeightOf(c) * target(b, c) / (p * static_cast<float>(batchSize));
            }
        }
        return grad;
    }

    std::string getLossName() const override { return "Weighted Categorical Cross-Entropy Loss"; }

    std::shared_ptr<owLoss> clone() const override {
        return std::make_shared<owWeightedCategoricalCrossEntropyLoss>(m_classWeights);
    }

private:
    float classWeightOf(size_t c) const {
        return c < m_classWeights.size() ? m_classWeights[c] : 1.0f;
    }

    std::vector<float> m_classWeights;
};

} // namespace ow
