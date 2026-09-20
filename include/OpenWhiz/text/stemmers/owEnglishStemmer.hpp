#pragma once

#include <string>

namespace ow {

// No-op passthrough: English inflection is light enough that suffix stripping
// typically isn't necessary for bag-of-words sentiment scoring. Kept as its own
// class (same shape as owTurkishStemmer/owFrenchStemmer) so owStemmer's
// dispatch is uniform and is the single place to add real stemming for
// English if a use case ever needs it.
class owEnglishStemmer {
public:
    std::string stem(const std::string& word) const {
        return word;
    }
};

} // namespace ow
