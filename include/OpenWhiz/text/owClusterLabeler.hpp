#pragma once

// Generic unsupervised clustering + human-readable labeling on top of
// arbitrary embedding vectors: takes fixed-size vectors (e.g. from
// owEmbeddingLookup::embedAverage()) plus each vector's stemmed tokens, and
// returns a cluster assignment per item plus each cluster's most frequent
// stems as a proposed label.
//
// Reuses owClusterLayer directly (centroid distances), trained with
// MSE-to-zero via owADAMOptimizer - the same objective as
// examples/clusterExample. No owDataset/CSV round-trip needed since
// owClusterLayer takes raw owTensor input.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// owADAMOptimizer.hpp must be included before owClusterLayer.hpp - the
// latter only forward-declares owOptimizer but calls m_optimizer->update()
// inline, which needs the complete type.
#include "../optimizers/owADAMOptimizer.hpp"
#include "../layers/owClusterLayer.hpp"
#include "../losses/owMeanSquaredErrorLoss.hpp"

namespace ow {

// Top stems + a proposed label for one cluster.
struct owClusterInfo {
    int clusterId = -1;
    int size = 0;
    // (stem, frequency) across all items assigned to this cluster, sorted by
    // frequency descending, truncated to Options::topStemCount.
    std::vector<std::pair<std::string, int>> topStems;
    // Convenience: topStems.front().first, or "(empty)" for a cluster with no
    // members (can happen — owClusterLayer's centroid count is fixed up front,
    // not every centroid necessarily wins any point).
    std::string suggestedLabel;
};

struct owClusterLabelResult {
    // assignments[i] is the cluster index (0..numClusters-1) for the i-th input item.
    std::vector<int> assignments;
    // One entry per cluster, indexed by clusterId.
    std::vector<owClusterInfo> clusters;
};

class owClusterLabeler {
public:
    struct Options {
        int numClusters = 8;
        int maxEpochs = 200;
        float learningRate = 0.05f;
        int topStemCount = 5;
    };

    // embeddings[i] and stemmedTokens[i] must correspond to the same item;
    // an empty stemmedTokens[i] just means that item won't contribute to any
    // cluster's stem frequencies.
    //
    // Split into two overloads instead of an `Options options = Options()`
    // default argument because clang rejects a nested struct's own default
    // member initializers as a default argument inside its enclosing class.
    static owClusterLabelResult cluster(const std::vector<std::vector<float>>& embeddings,
                                         const std::vector<std::vector<std::string>>& stemmedTokens) {
        return cluster(embeddings, stemmedTokens, Options());
    }

    static owClusterLabelResult cluster(const std::vector<std::vector<float>>& embeddings,
                                         const std::vector<std::vector<std::string>>& stemmedTokens,
                                         const Options& options) {
        if (embeddings.empty()) {
            throw std::runtime_error("owClusterLabeler::cluster: embeddings is empty");
        }
        if (embeddings.size() != stemmedTokens.size()) {
            throw std::runtime_error("owClusterLabeler::cluster: embeddings/stemmedTokens size mismatch");
        }
        if (options.numClusters <= 0) {
            throw std::runtime_error("owClusterLabeler::cluster: numClusters must be positive");
        }

        const size_t n = embeddings.size();
        const size_t dim = embeddings[0].size();
        const size_t k = static_cast<size_t>(options.numClusters);

        owTensor<float, 2> input(n, dim);
        for (size_t i = 0; i < n; ++i) {
            if (embeddings[i].size() != dim) {
                throw std::runtime_error("owClusterLabeler::cluster: inconsistent embedding dimension");
            }
            for (size_t d = 0; d < dim; ++d) {
                input(i, d) = embeddings[i][d];
            }
        }

        owClusterLayer layer(dim, k);
        owADAMOptimizer optimizer(options.learningRate);
        layer.setOptimizer(&optimizer);
        owMeanSquaredErrorLoss loss;

        owTensor<float, 2> target(n, k);
        target.setZero();

        for (int epoch = 0; epoch < options.maxEpochs; ++epoch) {
            owTensor<float, 2> output = layer.forward(input);
            owTensor<float, 2> grad = loss.gradient(output, target);
            layer.backward(grad);
            layer.train();
        }

        owTensor<float, 2> finalDistances = layer.forward(input);

        owClusterLabelResult result;
        result.assignments.resize(n);
        result.clusters.resize(k);
        for (size_t c = 0; c < k; ++c) {
            result.clusters[c].clusterId = static_cast<int>(c);
        }

        std::vector<std::unordered_map<std::string, int>> stemFreq(k);
        for (size_t i = 0; i < n; ++i) {
            size_t best = 0;
            float bestDist = finalDistances(i, 0);
            for (size_t c = 1; c < k; ++c) {
                float d = finalDistances(i, c);
                if (d < bestDist) {
                    bestDist = d;
                    best = c;
                }
            }
            result.assignments[i] = static_cast<int>(best);
            result.clusters[best].size++;
            for (const std::string& stem : stemmedTokens[i]) {
                stemFreq[best][stem]++;
            }
        }

        for (size_t c = 0; c < k; ++c) {
            std::vector<std::pair<std::string, int>> freqVec(stemFreq[c].begin(), stemFreq[c].end());
            std::sort(freqVec.begin(), freqVec.end(),
                      [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                          return a.second > b.second;
                      });
            if (static_cast<int>(freqVec.size()) > options.topStemCount) {
                freqVec.resize(options.topStemCount);
            }
            result.clusters[c].topStems = freqVec;
            result.clusters[c].suggestedLabel = freqVec.empty() ? "(empty)" : freqVec.front().first;
        }

        return result;
    }
};

}  // namespace ow
