#pragma once

// Generic TF-IDF vectorizer over already-tokenized documents
// (e.g. stemmed token lists from owStemmer/owTurkishStemmer). Fits a vocabulary +
// per-term IDF from a corpus, then returns one dense TF-IDF vector per document -
// a bag-of-words alternative to averaged pretrained embeddings
// (owEmbeddingLookup::embedAverage()), useful when averaging dilutes topic signal
// too much (short documents, no pretrained vectors for the domain, etc.). The
// resulting vectors are plain float vectors, so they plug directly into
// owClusterLabeler::cluster() the same way embedAverage() output does.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ow {

class owTfIdfVectorizer {
public:
    struct Options {
        // Terms appearing in fewer than this many documents are dropped - mirrors
        // the min-document-frequency pruning used for owEmbeddingLookup's
        // vocabulary (see owWordVectorPruner.hpp), same rationale:
        // singleton terms add dimensions without adding signal.
        int minDocFrequency = 2;
        // Terms appearing in MORE than this fraction of documents are dropped too -
        // e.g. 0.5 excludes anything present in over half the corpus. Ranking
        // candidates purely by "highest document frequency" (as maxVocabularySize
        // does below) otherwise systematically favors near-universal filler over
        // the lower-but-not-lowest-frequency terms that actually carry topic
        // signal - the more content-rich and lexically diverse the corpus, the
        // worse that bias gets. Default 1.0 applies no ceiling; set it
        // explicitly to enable this filter.
        float maxDocFrequencyRatio = 1.0f;
        // Vocabulary is capped to the N terms with the highest (but, after the
        // maxDocFrequencyRatio filter above, not too-high) document frequency -
        // keeps the output dimension bounded regardless of corpus size.
        int maxVocabularySize = 300;
        // sklearn-style: divide each document's vector by its L2 norm so document
        // length doesn't dominate the resulting cosine-like geometry that
        // owClusterLayer's Euclidean centroid distances rely on.
        bool l2Normalize = true;
    };

    // Split into two overloads (rather than a default `Options options =
    // Options()` argument) for the same reason as owClusterLabeler::cluster():
    // clang rejects a nested struct's own default member initializers when used
    // as a default argument inside the enclosing class.
    std::vector<std::vector<float>> fitTransform(const std::vector<std::vector<std::string>>& documents) {
        return fitTransform(documents, Options());
    }

    std::vector<std::vector<float>> fitTransform(const std::vector<std::vector<std::string>>& documents,
                                                  const Options& options) {
        if (documents.empty()) {
            throw std::runtime_error("owTfIdfVectorizer::fitTransform: documents is empty");
        }

        std::unordered_map<std::string, int> docFrequency;
        for (const std::vector<std::string>& doc : documents) {
            std::unordered_map<std::string, bool> seen;
            for (const std::string& term : doc) {
                seen[term] = true;
            }
            for (const auto& entry : seen) {
                docFrequency[entry.first]++;
            }
        }

        const int maxAllowedDf = static_cast<int>(options.maxDocFrequencyRatio * documents.size());
        std::vector<std::pair<std::string, int>> candidates;
        for (const auto& entry : docFrequency) {
            if (entry.second >= options.minDocFrequency && entry.second <= maxAllowedDf) {
                candidates.push_back(entry);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                      if (a.second != b.second) return a.second > b.second;
                      return a.first < b.first;  // stable tie-break, deterministic vocab order
                  });
        if (static_cast<int>(candidates.size()) > options.maxVocabularySize) {
            candidates.resize(options.maxVocabularySize);
        }

        m_vocab.clear();
        m_vocabIndex.clear();
        m_idf.assign(candidates.size(), 0.0f);
        const float n = static_cast<float>(documents.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            m_vocab.push_back(candidates[i].first);
            m_vocabIndex[candidates[i].first] = static_cast<int>(i);
            // Smoothed IDF (sklearn TfidfVectorizer default): stays finite and
            // positive even for a term present in every document.
            m_idf[i] = std::log((1.0f + n) / (1.0f + static_cast<float>(candidates[i].second))) + 1.0f;
        }

        return transform(documents, options.l2Normalize);
    }

    // Vectorizes new documents against an already-fit vocabulary/IDF (e.g. from a
    // prior fitTransform() call) - out-of-vocabulary terms are simply ignored.
    std::vector<std::vector<float>> transform(const std::vector<std::vector<std::string>>& documents,
                                               bool l2Normalize) const {
        std::vector<std::vector<float>> result;
        result.reserve(documents.size());
        for (const std::vector<std::string>& doc : documents) {
            std::vector<float> vec(m_vocab.size(), 0.0f);
            for (const std::string& term : doc) {
                auto it = m_vocabIndex.find(term);
                if (it == m_vocabIndex.end()) continue;
                vec[it->second] += 1.0f;
            }
            for (size_t i = 0; i < vec.size(); ++i) {
                vec[i] *= m_idf[i];
            }
            if (l2Normalize) {
                float norm = 0.0f;
                for (float v : vec) norm += v * v;
                norm = std::sqrt(norm);
                if (norm > 1e-8f) {
                    for (float& v : vec) v /= norm;
                }
            }
            result.push_back(std::move(vec));
        }
        return result;
    }

    const std::vector<std::string>& vocabulary() const { return m_vocab; }

private:
    std::vector<std::string> m_vocab;
    std::unordered_map<std::string, int> m_vocabIndex;
    std::vector<float> m_idf;
};

}  // namespace ow
