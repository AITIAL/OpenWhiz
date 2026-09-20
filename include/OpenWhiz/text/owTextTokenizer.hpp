#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "OpenWhiz/text/owLanguage.hpp"

namespace ow {

// Word-level tokenizer for short list titles/descriptions: splits on anything that
// isn't a letter or digit, lowercases with language-correct casing. UTF-8 aware
// throughout; does not depend on any C library locale setting.
//
// Language affects which accented letters count as word characters and how they
// lowercase:
// - English: plain ASCII only.
// - Turkish: ASCII + Turkish letters (Ç Ğ İ Ö Ş Ü and lowercase forms), with the
//   Turkish-specific ASCII "I" -> dotless "ı" [U+0131] and "İ" [U+0130] -> dotted
//   ASCII "i" mapping (generic ASCII case folding gets this wrong for Turkish).
// - French: ASCII + French accented letters (À Â Ä Ç É È Ê Ë Î Ï Ô Ö Ù Û Ü Ÿ Œ Æ
//   and lowercase forms), standard case folding (no special-case pairs needed).
class owTextTokenizer {
public:
    explicit owTextTokenizer(owLanguage language = owLanguage::English) : m_language(language) {}

    std::vector<std::string> tokenize(const std::string& text) const {
        std::vector<uint32_t> codepoints = utf8Decode(text);

        std::vector<std::string> tokens;
        std::string current;
        for (uint32_t cp : codepoints) {
            if (isWordCodepoint(cp)) {
                current += utf8Encode(toLower(cp));
            } else if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
        if (!current.empty()) {
            tokens.push_back(current);
        }
        return tokens;
    }

private:
    owLanguage m_language;

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
                // Invalid lead byte — skip it rather than misparse the rest of the string.
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

    static std::string utf8Encode(uint32_t cp) {
        std::string out;
        if (cp <= 0x7F) {
            out += static_cast<char>(cp);
        } else if (cp <= 0x7FF) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp <= 0xFFFF) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return out;
    }

    bool isWordCodepoint(uint32_t cp) const {
        if (cp >= '0' && cp <= '9') return true;
        if (cp >= 'a' && cp <= 'z') return true;
        if (cp >= 'A' && cp <= 'Z') return true;
        if (m_language == owLanguage::Turkish) {
            switch (cp) {
                case 0x00C7: case 0x00E7: // Ç ç
                case 0x011E: case 0x011F: // Ğ ğ
                case 0x0130: case 0x0131: // İ ı
                case 0x00D6: case 0x00F6: // Ö ö
                case 0x015E: case 0x015F: // Ş ş
                case 0x00DC: case 0x00FC: // Ü ü
                    return true;
                default:
                    return false;
            }
        }
        if (m_language == owLanguage::French) {
            switch (cp) {
                case 0x00C0: case 0x00E0: // À à
                case 0x00C2: case 0x00E2: // Â â
                case 0x00C4: case 0x00E4: // Ä ä
                case 0x00C7: case 0x00E7: // Ç ç
                case 0x00C8: case 0x00E8: // È è
                case 0x00C9: case 0x00E9: // É é
                case 0x00CA: case 0x00EA: // Ê ê
                case 0x00CB: case 0x00EB: // Ë ë
                case 0x00CE: case 0x00EE: // Î î
                case 0x00CF: case 0x00EF: // Ï ï
                case 0x00D4: case 0x00F4: // Ô ô
                case 0x00D6: case 0x00F6: // Ö ö
                case 0x00D9: case 0x00F9: // Ù ù
                case 0x00DB: case 0x00FB: // Û û
                case 0x00DC: case 0x00FC: // Ü ü
                case 0x0178: case 0x00FF: // Ÿ ÿ
                case 0x0152: case 0x0153: // Œ œ
                case 0x00C6: case 0x00E6: // Æ æ
                    return true;
                default:
                    return false;
            }
        }
        return false; // English: plain ASCII only.
    }

    uint32_t toLower(uint32_t cp) const {
        if (m_language == owLanguage::Turkish) return toLowerTurkish(cp);
        if (m_language == owLanguage::French) return toLowerFrench(cp);
        return toLowerAscii(cp);
    }

    static uint32_t toLowerAscii(uint32_t cp) {
        if (cp >= 'A' && cp <= 'Z') return cp + 0x20;
        return cp;
    }

    static uint32_t toLowerTurkish(uint32_t cp) {
        if (cp == 'I') return 0x0131;          // ASCII I -> dotless ı
        if (cp == 0x0130) return 'i';          // İ -> ASCII i
        if (cp >= 'A' && cp <= 'Z') return cp + 0x20;
        switch (cp) {
            case 0x00C7: return 0x00E7; // Ç -> ç
            case 0x011E: return 0x011F; // Ğ -> ğ
            case 0x00D6: return 0x00F6; // Ö -> ö
            case 0x015E: return 0x015F; // Ş -> ş
            case 0x00DC: return 0x00FC; // Ü -> ü
            default: return cp;
        }
    }

    static uint32_t toLowerFrench(uint32_t cp) {
        if (cp >= 'A' && cp <= 'Z') return cp + 0x20;
        switch (cp) {
            case 0x00C0: return 0x00E0; // À -> à
            case 0x00C2: return 0x00E2; // Â -> â
            case 0x00C4: return 0x00E4; // Ä -> ä
            case 0x00C7: return 0x00E7; // Ç -> ç
            case 0x00C8: return 0x00E8; // È -> è
            case 0x00C9: return 0x00E9; // É -> é
            case 0x00CA: return 0x00EA; // Ê -> ê
            case 0x00CB: return 0x00EB; // Ë -> ë
            case 0x00CE: return 0x00EE; // Î -> î
            case 0x00CF: return 0x00EF; // Ï -> ï
            case 0x00D4: return 0x00F4; // Ô -> ô
            case 0x00D6: return 0x00F6; // Ö -> ö
            case 0x00D9: return 0x00F9; // Ù -> ù
            case 0x00DB: return 0x00FB; // Û -> û
            case 0x00DC: return 0x00FC; // Ü -> ü
            case 0x0178: return 0x00FF; // Ÿ -> ÿ
            case 0x0152: return 0x0153; // Œ -> œ
            case 0x00C6: return 0x00E6; // Æ -> æ
            default: return cp;
        }
    }
};

} // namespace ow
