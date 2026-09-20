#pragma once

#include <string>

#include "OpenWhiz/text/owLanguage.hpp"
#include "OpenWhiz/text/stemmers/owTurkishStemmer.hpp"
#include "OpenWhiz/text/stemmers/owFrenchStemmer.hpp"
#include "OpenWhiz/text/stemmers/owEnglishStemmer.hpp"

namespace ow {

// Language-dispatching stemmer facade: routes stem() to the right per-language
// implementation.
class owStemmer {
public:
    explicit owStemmer(owLanguage language = owLanguage::English) : m_language(language) {}

    std::string stem(const std::string& word) const {
        switch (m_language) {
            case owLanguage::Turkish:
                return m_turkish.stem(word);
            case owLanguage::French:
                return m_french.stem(word);
            default:
                return m_english.stem(word);
        }
    }

private:
    owLanguage m_language;
    owTurkishStemmer m_turkish;
    owFrenchStemmer m_french;
    owEnglishStemmer m_english;
};

} // namespace ow
