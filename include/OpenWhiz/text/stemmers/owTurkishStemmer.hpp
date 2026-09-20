#pragma once

#include <string>
#include <vector>

namespace ow {

// Rule-based Turkish suffix stripping — no dictionary, no morphological analyzer,
// no external dependency. Iteratively strips the longest matching known nominal
// suffix (plural, possessive, case) from the end of a word, across a few passes,
// to approximate agglutinative stemming (e.g. "kitaplarimizdan" -> "kitap").
// Suffix stripping only: does not model vowel-drop irregularities (e.g.
// "burun" -> "burnu") or verb conjugation, and strips a known suffix from the
// end of a word whether or not it's genuinely an inflection (a short word
// that happens to end the same way is stripped too). Input is expected to
// already be lowercased tokens.
class owTurkishStemmer {
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
    static constexpr int kMaxPasses = 4;
    static constexpr int kMinRootLength = 2;

    // Ordered longest-suffix-first (in UTF-8 codepoints) so a more specific
    // suffix like "imiz" is tried before its shorter substring "iz"/"z".
    static const std::vector<std::string>& suffixes() {
        static const std::vector<std::string> table = {
            // possessive 1pl / 2pl (4 codepoints)
            "ımız", "imiz", "umuz", "ümüz",
            "ınız", "iniz", "unuz", "ünüz",
            // plural + possessive 3rd person (4 codepoints)
            "ları", "leri",
            // instrumental/"with" clitic with buffer consonant (3 codepoints)
            "yla", "yle",
            // genitive with buffer consonant "n" (3 codepoints)
            "nın", "nin", "nun", "nün",
            // ablative (3 codepoints)
            "dan", "den", "tan", "ten",
            // accusative with buffer consonant "y" (2 codepoints)
            "yı", "yi", "yu", "yü",
            // dative with buffer consonant "y" (2 codepoints)
            "ya", "ye",
            // possessive 3sg with buffer consonant "s" (2 codepoints)
            "sı", "si", "su", "sü",
            // plural (3 codepoints, byte-longer than most 2-codepoint entries above)
            "lar", "ler",
            // locative (2 codepoints)
            "da", "de", "ta", "te",
            // possessive 1sg / 2sg / genitive (2 codepoints)
            "ım", "im", "um", "üm",
            "ın", "in", "un", "ün",
            // instrumental/"with" clitic (2 codepoints)
            "la", "le",
            // Deliberately no 1-codepoint suffixes (bare "a","e","ı","i","u","ü" for
            // dative/accusative/genitive/possessive-3sg): they collide with the last
            // letter of many roots (e.g. "kedi" ends in "i") and cause over-stemming.
            // Trading recall on those specific inflections for precision elsewhere.
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

    // Counts UTF-8 codepoints (not bytes) — Turkish letters like i/s/g/u/o/c with
    // diacritics are 2-byte UTF-8 sequences, so byte length alone overcounts.
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
