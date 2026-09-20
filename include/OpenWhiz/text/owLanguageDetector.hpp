#pragma once

#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "OpenWhiz/text/owLanguage.hpp"

namespace ow {

// Heuristic per-text language detector for short, mixed-language user content
// (list titles/descriptions) - not a general-purpose language classifier, and
// not meant to be one. Produces the same owLanguage value consumed by
// owTextTokenizer, owStemmer, and owMultilingualEmbeddingLookup, so a single
// detection result can drive tokenization, stemming, and embedding-table
// selection consistently instead of each caller guessing separately.
//
// Detection order:
//  1. Turkish-exclusive letters (ı İ ğ Ğ ş Ş - never appear in English or
//     French) -> Turkish, unconditionally.
//  2. French-exclusive accented letters (à â ä é è ê ë î ï ô ù û ÿ œ æ, both
//     cases) -> French.
//  3. Letters shared between Turkish and French only (ç ö ü, both cases) ->
//     Turkish. A genuinely ambiguous case; resolved to one definite language
//     rather than left unresolved, since callers need a single owLanguage to
//     drive tokenization/stemming/embedding selection with.
//  4. No diacritics at all (plain ASCII) -> word-signal heuristic: count
//     hits against a small list of common Turkish vs. English function
//     words; the higher count wins, ties (including 0-0, e.g. a bare proper
//     noun) default to Turkish.
class owLanguageDetector {
public:
    static owLanguage detect(const std::string& text) {
        std::vector<uint32_t> codepoints = utf8Decode(text);

        bool hasTurkishExclusive = false;
        bool hasFrenchExclusive = false;
        bool hasSharedDiacritic = false;
        for (uint32_t cp : codepoints) {
            if (isTurkishExclusive(cp)) hasTurkishExclusive = true;
            else if (isFrenchExclusive(cp)) hasFrenchExclusive = true;
            else if (isSharedDiacritic(cp)) hasSharedDiacritic = true;
        }

        if (hasTurkishExclusive) return owLanguage::Turkish;
        if (hasFrenchExclusive) return owLanguage::French;
        if (hasSharedDiacritic) return owLanguage::Turkish;
        return detectFromWordSignals(text);
    }

private:
    static bool isTurkishExclusive(uint32_t cp) {
        switch (cp) {
            case 0x0131: case 0x0130: // ı İ
            case 0x011F: case 0x011E: // ğ Ğ
            case 0x015F: case 0x015E: // ş Ş
                return true;
            default:
                return false;
        }
    }

    static bool isFrenchExclusive(uint32_t cp) {
        switch (cp) {
            case 0x00E0: case 0x00C0: // à À
            case 0x00E2: case 0x00C2: // â Â
            case 0x00E4: case 0x00C4: // ä Ä
            case 0x00E9: case 0x00C9: // é É
            case 0x00E8: case 0x00C8: // è È
            case 0x00EA: case 0x00CA: // ê Ê
            case 0x00EB: case 0x00CB: // ë Ë
            case 0x00EE: case 0x00CE: // î Î
            case 0x00EF: case 0x00CF: // ï Ï
            case 0x00F4: case 0x00D4: // ô Ô
            case 0x00F9: case 0x00D9: // ù Ù
            case 0x00FB: case 0x00DB: // û Û
            case 0x00FF: case 0x0178: // ÿ Ÿ
            case 0x0153: case 0x0152: // œ Œ
            case 0x00E6: case 0x00C6: // æ Æ
                return true;
            default:
                return false;
        }
    }

    static bool isSharedDiacritic(uint32_t cp) {
        switch (cp) {
            case 0x00E7: case 0x00C7: // ç Ç
            case 0x00F6: case 0x00D6: // ö Ö
            case 0x00FC: case 0x00DC: // ü Ü
                return true;
            default:
                return false;
        }
    }

    static owLanguage detectFromWordSignals(const std::string& text) {
        // A small, hand-picked set of unambiguous, high-frequency words per
        // language - deliberately not an exhaustive dictionary. Common
        // function words plus a handful of frequent content words are enough
        // to disambiguate short, informal text (a few words to a sentence);
        // a general-purpose classifier's much larger vocabulary would be
        // overkill here and slower to check per word.
        static const std::unordered_set<std::string> kTurkishSignals = {
            "ve", "bir", "bu", "su", "en", "ile", "icin", "benim", "gibi",
            "kitaplarim", "filmler", "listem", "sevdigim", "favorim",
            "gezimiz", "seyahatimiz", "yemekler",
        };
        static const std::unordered_set<std::string> kEnglishSignals = {
            "the", "and", "my", "best", "favorite", "favourite", "of", "for",
            "list", "movies", "movie", "books", "book", "things", "diary",
            "is", "are", "was", "were", "with",
        };

        int turkishHits = 0, englishHits = 0;
        std::string word;
        for (size_t i = 0; i <= text.size(); ++i) {
            unsigned char c = (i < text.size()) ? static_cast<unsigned char>(text[i]) : 0;
            bool isAsciiAlnum = (c < 0x80) && std::isalnum(c);
            if (isAsciiAlnum) {
                word.push_back(static_cast<char>(std::tolower(c)));
            } else {
                if (!word.empty()) {
                    if (kTurkishSignals.count(word) > 0) ++turkishHits;
                    if (kEnglishSignals.count(word) > 0) ++englishHits;
                    word.clear();
                }
            }
        }

        return englishHits > turkishHits ? owLanguage::English : owLanguage::Turkish;
    }

    // Same minimal decoder shape as owTextTokenizer's - kept local rather than
    // shared since it's a few lines and this header has no other dependency
    // on the tokenizer.
    static std::vector<uint32_t> utf8Decode(const std::string& text) {
        std::vector<uint32_t> codepoints;
        size_t i = 0;
        while (i < text.size()) {
            unsigned char lead = static_cast<unsigned char>(text[i]);
            uint32_t cp = 0;
            size_t extra = 0;

            if ((lead & 0x80) == 0x00) {
                cp = lead;
                extra = 0;
            } else if ((lead & 0xE0) == 0xC0) {
                cp = lead & 0x1F;
                extra = 1;
            } else if ((lead & 0xF0) == 0xE0) {
                cp = lead & 0x0F;
                extra = 2;
            } else if ((lead & 0xF8) == 0xF0) {
                cp = lead & 0x07;
                extra = 3;
            } else {
                ++i;
                continue;
            }

            if (i + extra >= text.size()) {
                break;
            }

            bool valid = true;
            for (size_t j = 1; j <= extra; ++j) {
                unsigned char cont = static_cast<unsigned char>(text[i + j]);
                if ((cont & 0xC0) != 0x80) {
                    valid = false;
                    break;
                }
                cp = (cp << 6) | (cont & 0x3F);
            }

            if (!valid) {
                ++i;
                continue;
            }

            codepoints.push_back(cp);
            i += extra + 1;
        }
        return codepoints;
    }
};

} // namespace ow
