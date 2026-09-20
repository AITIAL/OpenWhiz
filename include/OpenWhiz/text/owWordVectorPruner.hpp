#pragma once

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace ow {

// Prunes a full pretrained word-vector table (several GB, hundreds of thousands
// to millions of words) down to just the words a project actually needs,
// producing a small fastText-.vec-compatible text table that
// owEmbeddingLookup::loadFromFile() can load at runtime. A library header
// rather than a standalone script, so it can be called directly from a
// project's own one-off data-prep code without a separate build/run step.
class owWordVectorPruner {
public:
    // Prunes a plain-text fastText/word2vec-text-format file: a "<count> <dim>"
    // header line followed by "word f1 f2 ... fN" lines. Returns the number of
    // words written, or -1 on failure to open either file.
    static long long pruneTextFormat(const std::string& inputPath,
                                      const std::unordered_set<std::string>& vocabulary,
                                      const std::string& outputPath) {
        std::ifstream in(inputPath);
        if (!in.is_open()) return -1;

        std::string headerLine;
        std::getline(in, headerLine);
        long long dim = 0;
        {
            std::istringstream hs(headerLine);
            long long declaredCount = 0;
            hs >> declaredCount >> dim;
        }

        std::ofstream out(outputPath);
        if (!out.is_open()) return -1;
        std::streampos headerPos = out.tellp();
        out << "0000000 " << dim << "\n";

        long long found = 0;
        std::string line;
        while (std::getline(in, line)) {
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            if (vocabulary.count(line.substr(0, sp))) {
                out << line << "\n";
                ++found;
            }
        }

        out.seekp(headerPos);
        char headerBuf[32];
        std::snprintf(headerBuf, sizeof(headerBuf), "%07lld", found);
        out << headerBuf;
        return found;
    }

    // Prunes a classic word2vec.c binary-format file: a "<count> <dim>\n" ASCII
    // header, then repeated "<word><space><dim x float32 LE bytes>" entries with
    // no reliable separator before the next word (a stray '\n' may appear and
    // must be skipped, not treated as a terminator - this matches the original
    // word2vec.c writer/reader behavior). Used by several major pretrained
    // word-vector releases distributed in this binary format, and any other
    // tool that emits it. Returns the number of words written, or -1 on
    // failure to open either file / parse the header.
    static long long pruneWord2VecBinaryFormat(const std::string& inputPath,
                                                const std::unordered_set<std::string>& vocabulary,
                                                const std::string& outputPath) {
        FILE* f = std::fopen(inputPath.c_str(), "rb");
        if (!f) return -1;

        long long wordCount = 0;
        int dim = 0;
        if (std::fscanf(f, "%lld %d", &wordCount, &dim) != 2) {
            std::fclose(f);
            return -1;
        }

        std::vector<std::pair<std::string, std::vector<float>>> foundEntries;
        std::string word;
        word.reserve(64);
        std::vector<float> vec(dim);

        for (long long i = 0; i < wordCount; ++i) {
            word.clear();
            int c = std::fgetc(f);
            while (c != EOF && (c == ' ' || c == '\n')) c = std::fgetc(f);
            while (c != EOF && c != ' ') {
                if (c != '\n') word.push_back((char)c);
                c = std::fgetc(f);
            }
            size_t got = std::fread(vec.data(), sizeof(float), (size_t)dim, f);
            if (got != (size_t)dim) break;
            if (vocabulary.count(word)) {
                foundEntries.emplace_back(word, vec);
            }
        }
        std::fclose(f);

        std::ofstream out(outputPath);
        if (!out.is_open()) return -1;
        out << foundEntries.size() << " " << dim << "\n";
        out << std::fixed;
        for (auto& entry : foundEntries) {
            out << entry.first;
            for (float v : entry.second) out << " " << std::setprecision(6) << v;
            out << "\n";
        }
        return (long long)foundEntries.size();
    }
};

} // namespace ow
