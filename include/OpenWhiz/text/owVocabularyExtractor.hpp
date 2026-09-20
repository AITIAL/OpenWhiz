#pragma once

#include <fstream>
#include <set>
#include <string>

#include "OpenWhiz/text/owLanguage.hpp"
#include "OpenWhiz/text/owTextTokenizer.hpp"

namespace ow {

// Builds a project's vocabulary file for owWordVectorPruner from a raw text
// corpus (one document/title per line), using the SAME tokenization rules as
// owTextTokenizer at runtime - building the vocabulary with different
// lowercasing/word-boundary rules than what runs at lookup time causes silent
// embedAverage() misses. Calling the real tokenizer here (rather than a
// separate reimplementation) guarantees the vocabulary always matches the
// exact rules embedAverage() will use at lookup time.
class owVocabularyExtractor {
public:
    static std::set<std::string> extractFromLines(const std::vector<std::string>& lines,
                                                    owLanguage language) {
        owTextTokenizer tokenizer(language);
        std::set<std::string> vocabulary;
        for (const std::string& line : lines) {
            for (const std::string& token : tokenizer.tokenize(line)) {
                vocabulary.insert(token);
            }
        }
        return vocabulary;
    }

    // One line per document/title in inputPath. Returns an empty set if the
    // file can't be opened.
    static std::set<std::string> extractFromFile(const std::string& inputPath,
                                                   owLanguage language) {
        std::ifstream in(inputPath);
        std::set<std::string> vocabulary;
        if (!in.is_open()) return vocabulary;

        owTextTokenizer tokenizer(language);
        std::string line;
        while (std::getline(in, line)) {
            for (const std::string& token : tokenizer.tokenize(line)) {
                vocabulary.insert(token);
            }
        }
        return vocabulary;
    }

    // Convenience: extract then write one word per line, sorted. Returns the
    // number of unique words written, or -1 if either file can't be opened.
    static long long extractToFile(const std::string& inputPath,
                                    const std::string& outputPath,
                                    owLanguage language) {
        std::ifstream in(inputPath);
        if (!in.is_open()) return -1;
        std::set<std::string> vocabulary = extractFromFile(inputPath, language);

        std::ofstream out(outputPath);
        if (!out.is_open()) return -1;
        for (const std::string& word : vocabulary) out << word << "\n";
        return (long long)vocabulary.size();
    }
};

} // namespace ow
