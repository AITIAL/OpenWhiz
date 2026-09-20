#pragma once

#include <string>
#include <vector>

namespace ow {

// Rule-based French suffix stripping — no dictionary, no morphological analyzer,
// no external dependency. Deliberately minimal: French inflection is far more
// limited than Turkish's agglutination (plural is mostly a single trailing "-s"/
// "-x", feminine mostly a trailing "-e", verbs conjugate via a handful of common
// endings), so its suffix table is intentionally smaller than
// owTurkishStemmer's. Does not model irregular plurals (e.g. "cheval" ->
// "chevaux"), verb stem changes, or elision. Input is expected to already be
// lowercased tokens.
class owFrenchStemmer {
public:
    std::string stem(const std::string& word) const {
        std::string current = word;
        for (int pass = 0; pass < kMaxPasses; ++pass) {
            const std::string* match = longestMatchingSuffix(current);
            if (match == nullptr) {
                break;
            }
            std::string stripped = current.substr(0, current.size() - match->size());
            if (utf8Length(stripped) < kMinRootLength) {
                break;
            }
            current = stripped;
        }
        return current;
    }

private:
    static constexpr int kMaxPasses = 2;
    static constexpr int kMinRootLength = 2;

    // Ordered longest-suffix-first (in UTF-8 codepoints): a few common verb
    // endings, then the plural markers ("s"/"x"), then the feminine marker ("e").
    static const std::vector<std::string>& suffixes() {
        static const std::vector<std::string> table = {
            // common verb endings (3 codepoints)
            "ent", "ons",
            // infinitive / common verb endings (2 codepoints)
            "er", "ir", "ez",
            // plural (1 codepoint)
            "x", "s",
            // feminine marker (1 codepoint)
            "e",
        };
        return table;
    }

    static const std::string* longestMatchingSuffix(const std::string& word) {
        const std::string* best = nullptr;
        for (const std::string& suffix : suffixes()) {
            if (word.size() <= suffix.size()) {
                continue;
            }
            if (word.compare(word.size() - suffix.size(), suffix.size(), suffix) == 0) {
                if (best == nullptr || suffix.size() > best->size()) {
                    best = &suffix;
                }
            }
        }
        return best;
    }

    // Counts UTF-8 codepoints (not bytes) — French letters like é/è/ê are 2-byte
    // UTF-8 sequences, so byte length alone overcounts.
    static int utf8Length(const std::string& s) {
        int length = 0;
        for (unsigned char c : s) {
            if ((c & 0xC0) != 0x80) {
                ++length;
            }
        }
        return length;
    }
};

} // namespace ow
