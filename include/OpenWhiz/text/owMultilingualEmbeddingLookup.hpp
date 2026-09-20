#pragma once

#include <string>
#include <vector>

#include "OpenWhiz/text/owEmbeddingLookup.hpp"
#include "OpenWhiz/text/owLanguage.hpp"

namespace ow {

// Language-dispatching facade over owEmbeddingLookup: holds one word-vector
// table per owLanguage and routes embedAverage()/tryGetVector() to the right
// one - same shape as owStemmer's dispatch over per-language stemmers.
//
// Word-vector tables for different languages are separately trained,
// unaligned vector spaces (true of fastText and word2vec-family tables
// alike) - a Turkish embedding and an English embedding are not comparable
// by cosine similarity or any other distance, even at the same dimension.
// This class exists so callers can't accidentally mix them: each table is
// only reachable through its own owLanguage, there is no "compare across
// languages" method here on purpose.
class owMultilingualEmbeddingLookup {
public:
    // Loads the word-vector table for one language. Safe to call for a
    // subset of owLanguage's values - embedAverage()/tryGetVector() for a
    // language whose table was never loaded just behave like an empty table
    // (all-zero vector, no match), same as a fresh unloaded owEmbeddingLookup.
    bool loadForLanguage(owLanguage language, const std::string& path) {
        return lookupFor(language).loadFromFile(path);
    }

    bool isLoaded(owLanguage language) const {
        return lookupFor(language).getDimension() > 0;
    }

    const owEmbeddingLookup& getLookup(owLanguage language) const {
        return lookupFor(language);
    }

    bool tryGetVector(const std::string& word, owLanguage language, std::vector<float>& out) const {
        return lookupFor(language).tryGetVector(word, out);
    }

    std::vector<float> embedAverage(const std::vector<std::string>& tokens, owLanguage language) const {
        return lookupFor(language).embedAverage(tokens);
    }

private:
    owEmbeddingLookup& lookupFor(owLanguage language) {
        switch (language) {
            case owLanguage::Turkish: return m_turkish;
            case owLanguage::French: return m_french;
            default: return m_english;
        }
    }

    const owEmbeddingLookup& lookupFor(owLanguage language) const {
        switch (language) {
            case owLanguage::Turkish: return m_turkish;
            case owLanguage::French: return m_french;
            default: return m_english;
        }
    }

    owEmbeddingLookup m_english;
    owEmbeddingLookup m_turkish;
    owEmbeddingLookup m_french;
};

} // namespace ow
