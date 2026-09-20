#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ow {

// Loads a pretrained fastText-format word vector table (word + N floats per line,
// optional "<count> <dim>" header line) into an in-memory lookup. Intended to be
// used with a table already pruned down to a project's actual vocabulary —
// see owWordVectorPruner.hpp — not the full multi-GB release, which would be
// wasteful to ship and load whole.
class owEmbeddingLookup {
public:
    bool loadFromFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;
        }

        m_vectors.clear();
        m_dimension = 0;

        std::string line;
        if (!std::getline(file, line)) {
            return false;
        }

        // fastText .vec files start with a "<vocab_size> <dim>" header line.
        // If the first line doesn't parse as that, treat it as a data line instead.
        std::istringstream headerStream(line);
        size_t declaredCount = 0;
        size_t declaredDim = 0;
        bool hasHeader = static_cast<bool>(headerStream >> declaredCount >> declaredDim)
                          && headerStream.eof();

        if (!hasHeader) {
            if (!parseLine(line)) {
                return false;
            }
        }

        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            if (!parseLine(line)) {
                return false;
            }
        }

        return m_dimension > 0;
    }

    bool tryGetVector(const std::string& word, std::vector<float>& out) const {
        auto it = m_vectors.find(word);
        if (it == m_vectors.end()) {
            return false;
        }
        out = it->second;
        return true;
    }

    bool contains(const std::string& word) const {
        return m_vectors.find(word) != m_vectors.end();
    }

    size_t getDimension() const {
        return m_dimension;
    }

    size_t getVocabularySize() const {
        return m_vectors.size();
    }

    // Mean-pools the vectors of known words in `tokens`, silently skipping
    // out-of-vocabulary tokens. Returns a zero vector if none of the tokens
    // are known or the table hasn't been loaded yet.
    std::vector<float> embedAverage(const std::vector<std::string>& tokens) const {
        std::vector<float> sum(m_dimension, 0.0f);
        size_t known = 0;
        for (const std::string& token : tokens) {
            auto it = m_vectors.find(token);
            if (it == m_vectors.end()) {
                continue;
            }
            for (size_t i = 0; i < m_dimension; ++i) {
                sum[i] += it->second[i];
            }
            ++known;
        }
        if (known > 0) {
            for (float& value : sum) {
                value /= static_cast<float>(known);
            }
        }
        return sum;
    }

    // Loads per-word weights (e.g. IDF) from a "word\tweight" file, one per line —
    // computing such a table requires document frequencies from your own corpus
    // (project-specific, not shipped here; write a small script that counts how
    // many documents each word appears in and derives a weight from that).
    // Independent of loadFromFile(); a word missing from this table just falls back
    // to weight 1.0 in embedWeightedAverage(), it isn't required to cover every word
    // in the vector table.
    bool loadWeights(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;
        }

        m_weights.clear();
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            std::string word = line.substr(0, tab);
            float weight = 0.0f;
            try {
                weight = std::stof(line.substr(tab + 1));
            } catch (...) {
                continue;
            }
            m_weights[word] = weight;
        }
        return true;
    }

    // Same as embedAverage(), but each known token's vector is scaled by its weight
    // (loaded via loadWeights(), e.g. corpus IDF) before averaging — tokens missing
    // from the weights table default to weight 1.0, same as a plain average would
    // give them. Returns a zero vector if no token is known or total weight is zero.
    std::vector<float> embedWeightedAverage(const std::vector<std::string>& tokens) const {
        std::vector<float> sum(m_dimension, 0.0f);
        float totalWeight = 0.0f;
        for (const std::string& token : tokens) {
            auto it = m_vectors.find(token);
            if (it == m_vectors.end()) {
                continue;
            }
            float weight = 1.0f;
            auto wit = m_weights.find(token);
            if (wit != m_weights.end()) {
                weight = wit->second;
            }
            for (size_t i = 0; i < m_dimension; ++i) {
                sum[i] += it->second[i] * weight;
            }
            totalWeight += weight;
        }
        if (totalWeight > 0.0f) {
            for (float& value : sum) {
                value /= totalWeight;
            }
        }
        return sum;
    }

private:
    bool parseLine(const std::string& line) {
        std::istringstream stream(line);
        std::string word;
        if (!(stream >> word)) {
            return false;
        }

        std::vector<float> values;
        float value = 0.0f;
        while (stream >> value) {
            values.push_back(value);
        }
        if (values.empty()) {
            return false;
        }

        if (m_dimension == 0) {
            m_dimension = values.size();
        } else if (values.size() != m_dimension) {
            return false;
        }

        m_vectors.emplace(std::move(word), std::move(values));
        return true;
    }

    std::unordered_map<std::string, std::vector<float>> m_vectors;
    std::unordered_map<std::string, float> m_weights;
    size_t m_dimension = 0;
};

} // namespace ow
