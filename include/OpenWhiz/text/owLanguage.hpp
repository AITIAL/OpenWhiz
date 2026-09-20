#pragma once

namespace ow {

// Language selector shared by owTextTokenizer and owStemmer. Default is English
// (no special-casing needed - plain ASCII word/case rules, no-op stemming).
enum class owLanguage {
    English,
    Turkish,
    French
};

} // namespace ow
