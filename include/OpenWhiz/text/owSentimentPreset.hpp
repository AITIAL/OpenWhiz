#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "OpenWhiz/openwhiz.hpp"
#include "OpenWhiz/losses/owWeightedCategoricalCrossEntropyLoss.hpp"
#include "OpenWhiz/losses/owWeightedBinaryCrossEntropyLoss.hpp"

namespace ow {

// Sentiment classification head built on OpenWhiz's existing pieces: 2-class
// uses owSigmoidActivation + owBinaryCrossEntropyLoss, 3+ classes use the
// softmax-style owProbabilityLayer + owCategoricalCrossEntropyLoss - same
// pattern as OpenWhiz's own classificationExample. Takes embedding vectors
// (e.g. from owEmbeddingLookup::embedAverage()) as input.
//
// owDataset has no public API to inject a numeric matrix directly, so
// train() writes embeddings+labels to a temp CSV and loads them through
// loadFromCSV() - the same data path OpenWhiz's own examples use.
class owSentimentPreset {
public:
    struct Options {
        std::vector<int> hiddenSizes = {32, 16};
        int maxEpochs = 300;
        bool autoNormalize = false;
        bool enablePrinting = false;
        // owDataset defaults to a 0.6/0.2/0.2 train/val/test split — fine for a real
        // corpus, but on small datasets it can starve the training set (e.g. 8 samples
        // -> only ~4 actually used for training). Adjust for your data size.
        float trainRatio = 0.6f;
        float valRatio = 0.2f;
        float testRatio = 0.2f;
        // owDataset::setRatios() shuffles row->split assignment by default.
        // Set this false for a stratified split where rows are pre-ordered
        // by the caller (first trainRatio fraction training, etc.).
        bool shuffleSplit = true;
        // Balanced class weighting for the numClasses>=3 (categorical) path only:
        // weight_c = totalExamples / (numClasses * count_c), the standard "balanced"
        // formula (matches sklearn's class_weight='balanced'). Computed from the full
        // label set passed to train() (not train-split-only, to stay independent of
        // shuffleSplit/row-ordering assumptions). Not oversampling - no rows are
        // duplicated, only the loss's per-example weight changes.
        bool useClassWeights = false;
    };

    // embeddings.size() == labels.size(); labels are class indices in [0, numClasses).
    // numClasses == 2 -> binary sigmoid+BCE head. numClasses >= 3 -> softmax+categorical
    // cross-entropy head. tempCsvPath is overwritten with the training data (numeric
    // only, no PII beyond what's already in the embedding vectors).
    bool train(const std::vector<std::vector<float>>& embeddings,
               const std::vector<int>& labels,
               int numClasses,
               const std::string& tempCsvPath) {
        return train(embeddings, labels, numClasses, tempCsvPath, Options());
    }

    bool train(const std::vector<std::vector<float>>& embeddings,
               const std::vector<int>& labels,
               int numClasses,
               const std::string& tempCsvPath,
               const Options& options) {
        return train(embeddings, labels, numClasses, tempCsvPath, options, nullptr);
    }

    // perExampleWeights (2-class/binary path only): one weight per row in
    // `embeddings`/`labels`, in the SAME order those were passed in. Only the
    // first N entries (N = the actual training-split row count owDataset ends
    // up with) are used, since the loss is only ever evaluated on the training
    // split - matches how useClassWeights below reads the FULL label set (not
    // train-split-only) but the loss itself is a training-time quantity.
    // Takes priority over options.useClassWeights when non-null and non-empty
    // (the two are alternative ways of shaping the same loss, not additive).
    bool train(const std::vector<std::vector<float>>& embeddings,
               const std::vector<int>& labels,
               int numClasses,
               const std::string& tempCsvPath,
               const Options& options,
               const std::vector<float>* perExampleWeights) {
        if (embeddings.empty() || embeddings.size() != labels.size() || numClasses < 2) {
            return false;
        }
        if (!writeTrainingCSV(tempCsvPath, embeddings, labels, numClasses)) {
            return false;
        }

        m_network = owNeuralNetwork();
        m_network.getDataset()->setAutoNormalizeEnabled(options.autoNormalize);
        if (!m_network.loadData(tempCsvPath)) {
            return false;
        }
        m_network.getDataset()->setTargetVariableNum(numClasses == 2 ? 1 : numClasses);
        m_network.getDataset()->setRatios(options.trainRatio, options.valRatio, options.testRatio, options.shuffleSplit);

        if (numClasses == 2) {
            m_network.createNeuralNetwork(options.hiddenSizes, "ReLU", "Sigmoid", false);
            if (perExampleWeights && !perExampleWeights->empty()) {
                size_t trainCount = m_network.getDataset()->getTrainInput().shape()[0];
                std::vector<float> trainWeights;
                trainWeights.reserve(trainCount);
                for (size_t i = 0; i < trainCount && i < perExampleWeights->size(); ++i) {
                    trainWeights.push_back((*perExampleWeights)[i]);
                }
                m_network.setLoss(std::make_shared<owWeightedBinaryCrossEntropyLoss>(std::move(trainWeights)));
            } else if (options.useClassWeights) {
                int countPos = 0, countNeg = 0;
                for (int lbl : labels) (lbl == 1 ? countPos : countNeg)++;
                float n = static_cast<float>(labels.size());
                float wPos = countPos > 0 ? n / (2.0f * countPos) : 1.0f;
                float wNeg = countNeg > 0 ? n / (2.0f * countNeg) : 1.0f;
                m_network.setLoss(std::make_shared<owWeightedBinaryCrossEntropyLoss>(wPos, wNeg));
            } else {
                m_network.setLoss(std::make_shared<owBinaryCrossEntropyLoss>());
            }
        } else {
            m_network.createNeuralNetwork(options.hiddenSizes, "ReLU", "Identity", false);
            m_network.addLayer(std::make_shared<owProbabilityLayer>());
            if (options.useClassWeights) {
                std::vector<int> counts(numClasses, 0);
                for (int lbl : labels) if (lbl >= 0 && lbl < numClasses) counts[lbl]++;
                std::vector<float> weights(numClasses, 1.0f);
                for (int c = 0; c < numClasses; ++c) {
                    weights[c] = counts[c] > 0
                        ? static_cast<float>(labels.size()) / (numClasses * static_cast<float>(counts[c]))
                        : 1.0f;
                }
                m_network.setLoss(std::make_shared<owWeightedCategoricalCrossEntropyLoss>(weights));
            } else {
                m_network.setLoss(std::make_shared<owCategoricalCrossEntropyLoss>());
            }
        }

        // owLBFGSOptimizer's line search can stall after one step on non-toy
        // data (its step size collapses and does not recover), so Adam is
        // used here instead.
        m_network.setOptimizer(std::make_shared<owADAMOptimizer>());
        m_network.setMaximumEpochNum(options.maxEpochs);
        m_network.setEnablePrinting(options.enablePrinting);
        m_network.train();
        m_numClasses = numClasses;
        m_lastFinishReason = m_network.getTrainingFinishReason();
        m_lastTrainLoss = m_network.getLastTrainError();
        return true;
    }

    // Multi-label variant: each row may belong to any number of the numLabels
    // classes at once (independent sigmoid per label, not softmax).
    // multiHotLabels[i][c] is 1 if row i has label c, 0 otherwise.
    bool trainMultiLabel(const std::vector<std::vector<float>>& embeddings,
                          const std::vector<std::vector<int>>& multiHotLabels,
                          int numLabels,
                          const std::string& tempCsvPath) {
        return trainMultiLabel(embeddings, multiHotLabels, numLabels, tempCsvPath, Options());
    }

    bool trainMultiLabel(const std::vector<std::vector<float>>& embeddings,
                          const std::vector<std::vector<int>>& multiHotLabels,
                          int numLabels,
                          const std::string& tempCsvPath,
                          const Options& options) {
        if (embeddings.empty() || embeddings.size() != multiHotLabels.size() || numLabels < 1) {
            return false;
        }
        if (!writeMultiLabelTrainingCSV(tempCsvPath, embeddings, multiHotLabels, numLabels)) {
            return false;
        }

        m_network = owNeuralNetwork();
        m_network.getDataset()->setAutoNormalizeEnabled(options.autoNormalize);
        if (!m_network.loadData(tempCsvPath)) {
            return false;
        }
        m_network.getDataset()->setTargetVariableNum(numLabels);
        m_network.getDataset()->setRatios(options.trainRatio, options.valRatio, options.testRatio, options.shuffleSplit);

        m_network.createNeuralNetwork(options.hiddenSizes, "ReLU", "Sigmoid", false);
        if (options.useClassWeights) {
            // Balanced weight per label, computed from that label's own
            // positive/negative count (not pooled across labels).
            std::vector<int> countPos(numLabels, 0), countNeg(numLabels, 0);
            for (const std::vector<int>& row : multiHotLabels) {
                for (int c = 0; c < numLabels; ++c) (row[c] != 0 ? countPos[c] : countNeg[c])++;
            }
            float n = static_cast<float>(multiHotLabels.size());
            std::vector<std::pair<float, float>> perLabelWeights(numLabels);
            for (int c = 0; c < numLabels; ++c) {
                float wPos = countPos[c] > 0 ? n / (2.0f * countPos[c]) : 1.0f;
                float wNeg = countNeg[c] > 0 ? n / (2.0f * countNeg[c]) : 1.0f;
                perLabelWeights[c] = {wPos, wNeg};
            }
            m_network.setLoss(std::make_shared<owWeightedBinaryCrossEntropyLoss>(perLabelWeights, static_cast<size_t>(numLabels)));
        } else {
            m_network.setLoss(std::make_shared<owBinaryCrossEntropyLoss>());
        }

        m_network.setOptimizer(std::make_shared<owADAMOptimizer>());
        m_network.setMaximumEpochNum(options.maxEpochs);
        m_network.setEnablePrinting(options.enablePrinting);
        m_network.train();
        m_numClasses = numLabels;
        m_lastFinishReason = m_network.getTrainingFinishReason();
        m_lastTrainLoss = m_network.getLastTrainError();
        return true;
    }

    // Diagnostics captured during the most recent train() call.
    std::string getLastFinishReason() const { return m_lastFinishReason; }
    float getLastTrainLoss() const { return m_lastTrainLoss; }

    // Returns per-class scores (2 values for binary [P(negative), P(positive)] derived
    // from the single sigmoid output, or numClasses softmax probabilities). Empty on
    // failure (e.g. train() wasn't called, or dimension mismatch).
    std::vector<float> predict(const std::vector<float>& embedding) {
        if (m_numClasses == 0 || embedding.empty()) {
            return {};
        }

        owTensor<float, 2> input(1, embedding.size());
        for (size_t i = 0; i < embedding.size(); ++i) {
            input(0, i) = embedding[i];
        }
        owTensor<float, 2> out = m_network.forward(input);

        if (m_numClasses == 2) {
            float positive = out(0, 0);
            return {1.0f - positive, positive};
        }

        std::vector<float> scores(m_numClasses);
        for (int c = 0; c < m_numClasses; ++c) {
            scores[c] = out(0, static_cast<size_t>(c));
        }
        return scores;
    }

private:
    bool writeTrainingCSV(const std::string& path,
                           const std::vector<std::vector<float>>& embeddings,
                           const std::vector<int>& labels,
                           int numClasses) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }
        // owDataset's numeric-column detector doesn't recognize scientific
        // notation ('e'/'E') - one such value marks the whole column as
        // Text instead of numeric. Force fixed notation so embedding floats
        // never trigger it.
        file << std::fixed << std::setprecision(9);

        size_t dim = embeddings.front().size();
        for (size_t d = 0; d < dim; ++d) {
            file << "e" << d << ",";
        }
        if (numClasses == 2) {
            file << "label\n";
        } else {
            for (int c = 0; c < numClasses; ++c) {
                file << "t" << c << (c + 1 < numClasses ? "," : "\n");
            }
        }

        for (size_t i = 0; i < embeddings.size(); ++i) {
            if (embeddings[i].size() != dim) {
                return false;
            }
            for (float v : embeddings[i]) {
                file << v << ",";
            }
            if (numClasses == 2) {
                file << (labels[i] == 1 ? "1" : "0") << "\n";
            } else {
                for (int c = 0; c < numClasses; ++c) {
                    file << (labels[i] == c ? "1" : "0") << (c + 1 < numClasses ? "," : "\n");
                }
            }
        }
        return true;
    }

    bool writeMultiLabelTrainingCSV(const std::string& path,
                                     const std::vector<std::vector<float>>& embeddings,
                                     const std::vector<std::vector<int>>& multiHotLabels,
                                     int numLabels) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }
        file << std::fixed << std::setprecision(9);

        size_t dim = embeddings.front().size();
        for (size_t d = 0; d < dim; ++d) {
            file << "e" << d << ",";
        }
        for (int c = 0; c < numLabels; ++c) {
            file << "t" << c << (c + 1 < numLabels ? "," : "\n");
        }

        for (size_t i = 0; i < embeddings.size(); ++i) {
            if (embeddings[i].size() != dim || multiHotLabels[i].size() != static_cast<size_t>(numLabels)) {
                return false;
            }
            for (float v : embeddings[i]) {
                file << v << ",";
            }
            for (int c = 0; c < numLabels; ++c) {
                file << (multiHotLabels[i][c] != 0 ? "1" : "0") << (c + 1 < numLabels ? "," : "\n");
            }
        }
        return true;
    }

    owNeuralNetwork m_network;
    int m_numClasses = 0;
    std::string m_lastFinishReason;
    float m_lastTrainLoss = 0.0f;
};

} // namespace ow
