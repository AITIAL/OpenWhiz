#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "OpenWhiz/openwhiz.hpp"

namespace ow {

// Class-weighted binary cross-entropy, same pattern as
// owWeightedCategoricalCrossEntropyLoss - a general-purpose loss for
// imbalanced binary/multi-label classification, not specific to any one
// domain. eps matches ow::owBinaryCrossEntropyLoss's own clamp (1e-12); no
// owProbabilityLayer is involved on the binary path (Sigmoid output feeds
// this loss directly), so there's no clamp-mismatch risk here like the
// softmax path had.
class owWeightedBinaryCrossEntropyLoss : public owLoss {
public:
    // weightPositive/weightNegative: per-class multiplier, e.g. balanced =
    // N / (2 * count_c). Applied to every example alike.
    owWeightedBinaryCrossEntropyLoss(float weightPositive, float weightNegative)
        : m_weightPositive(weightPositive), m_weightNegative(weightNegative) {}

    // Per-example weights - one entry per training row, in the exact row
    // order owNeuralNetwork::runStandardTrainingLoop() evaluates them in
    // (full-batch, fixed order, no per-epoch shuffling - see
    // owNeuralNetwork.inl's train()/runStandardTrainingLoop()). Useful when
    // the data mixes several subgroups (e.g. different sources, categories,
    // or languages) whose class ratios differ: a single global weight per
    // class can be wrong for every subgroup at once, so each row's weight
    // can instead be looked up per (subgroup, label) at the call site rather
    // than coming from one pooled ratio. A size mismatch against the actual
    // batch (shouldn't happen under normal use) falls back to a neutral
    // weight of 1.0 for the out-of-range rows rather than reading out of
    // bounds.
    explicit owWeightedBinaryCrossEntropyLoss(std::vector<float> perExampleWeights)
        : m_weightPositive(1.0f), m_weightNegative(1.0f), m_perExampleWeights(std::move(perExampleWeights)) {}

    // Per-label weights: one (weightPositive, weightNegative) pair per output
    // column, looked up by column index (flat tensor index modulo numLabels).
    // For multi-label targets where each column has its own class imbalance.
    owWeightedBinaryCrossEntropyLoss(std::vector<std::pair<float, float>> perLabelWeights, size_t numLabels)
        : m_weightPositive(1.0f), m_weightNegative(1.0f),
          m_perLabelWeights(std::move(perLabelWeights)), m_numLabels(numLabels) {}

    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t n = prediction.size();
        const float eps = 1e-12f;
        for (size_t i = 0; i < n; ++i) {
            float p = std::max(eps, std::min(1.0f - eps, prediction.data()[i]));
            float t = target.data()[i];
            float w = weightFor(i, t);
            loss -= w * (t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
        }
        return loss / static_cast<float>(n);
    }

    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.size();
        const float eps = 1e-12f;
        float factor = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            float p = std::max(eps, std::min(1.0f - eps, prediction.data()[i]));
            float t = target.data()[i];
            float w = weightFor(i, t);
            grad.data()[i] = factor * w * (-(t / p) + (1.0f - t) / (1.0f - p));
        }
        return grad;
    }

    std::string getLossName() const override { return "Weighted Binary Cross-Entropy Loss"; }

    std::shared_ptr<owLoss> clone() const override {
        if (!m_perLabelWeights.empty()) {
            return std::make_shared<owWeightedBinaryCrossEntropyLoss>(m_perLabelWeights, m_numLabels);
        }
        if (!m_perExampleWeights.empty()) {
            return std::make_shared<owWeightedBinaryCrossEntropyLoss>(m_perExampleWeights);
        }
        return std::make_shared<owWeightedBinaryCrossEntropyLoss>(m_weightPositive, m_weightNegative);
    }

private:
    float weightFor(size_t i, float t) const {
        if (!m_perLabelWeights.empty()) {
            size_t col = m_numLabels > 0 ? (i % m_numLabels) : 0;
            if (col >= m_perLabelWeights.size()) return 1.0f;
            return t > 0.5f ? m_perLabelWeights[col].first : m_perLabelWeights[col].second;
        }
        if (i < m_perExampleWeights.size()) return m_perExampleWeights[i];
        if (!m_perExampleWeights.empty()) return 1.0f; // size-mismatch fallback, see ctor doc comment
        return t > 0.5f ? m_weightPositive : m_weightNegative;
    }

    float m_weightPositive;
    float m_weightNegative;
    std::vector<float> m_perExampleWeights;
    std::vector<std::pair<float, float>> m_perLabelWeights;
    size_t m_numLabels = 0;
};

} // namespace ow
