#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "OpenWhiz/text/owEmbeddingLookup.hpp"
#include "OpenWhiz/text/owVocabularyExtractor.hpp"
#include "OpenWhiz/text/owWordVectorPruner.hpp"

// OpenWhiz/text pipeline demo: build a small vocabulary from a sample corpus
// with owVocabularyExtractor, prune a real pretrained word2vec-binary-format
// release down to just those words with owWordVectorPruner, then load the
// small pruned table with owEmbeddingLookup and print a few real vectors.
// Unlike sentimentClassificationExample/categoryClassificationExample/
// multiLabelExample (tiny hand-written synthetic vectors), this example
// expects a real pretrained vector file the user downloads separately - see
// README.md for where to get one. If that file isn't present, this prints
// instructions and exits cleanly instead of crashing.

static const char* kRawVectorsPath = "examples/vectorPruningExample/GoogleNews-vectors-negative300.bin";
static const char* kPrunedVectorsPath = "examples/vectorPruningExample/pruned.vec";

int main() {
    std::cout << "=== OpenWhiz/text: extract vocabulary -> prune real pretrained vectors -> load ===\n";

    std::vector<std::string> corpus = {
        "I love watching movies and listening to music",
        "My favorite book is a mystery novel",
        "The weather today is sunny and warm",
        "She bought a new computer for her office",
        "We traveled to Paris last summer",
    };
    std::set<std::string> vocabularySet =
        ow::owVocabularyExtractor::extractFromLines(corpus, ow::owLanguage::English);
    std::unordered_set<std::string> vocabulary(vocabularySet.begin(), vocabularySet.end());
    std::cout << "Extracted " << vocabulary.size() << " unique words from " << corpus.size()
               << " sample sentences.\n";

    std::ifstream rawCheck(kRawVectorsPath, std::ios::binary);
    if (!rawCheck.is_open()) {
        std::cout << "\nRaw pretrained vector file not found at \"" << kRawVectorsPath << "\".\n"
                   << "Download it (see this example's README.md for a link) and place it at that\n"
                   << "exact path, then re-run this example.\n";
        return 0;
    }
    rawCheck.close();

    long long found =
        ow::owWordVectorPruner::pruneWord2VecBinaryFormat(kRawVectorsPath, vocabulary, kPrunedVectorsPath);
    if (found < 0) {
        std::cout << "Failed to read \"" << kRawVectorsPath << "\" - is it a valid word2vec binary file?\n";
        return 1;
    }
    std::cout << "Pruned " << found << "/" << vocabulary.size()
               << " vocabulary words out of the full release into \"" << kPrunedVectorsPath << "\".\n";

    ow::owEmbeddingLookup lookup;
    if (!lookup.loadFromFile(kPrunedVectorsPath)) {
        std::cout << "Failed to load pruned vector table.\n";
        return 1;
    }
    std::cout << "Loaded " << lookup.getVocabularySize() << " words, dimension=" << lookup.getDimension()
               << ".\n\n";

    std::vector<std::string> sampleWords = {"music", "computer", "weather", "book"};
    for (const std::string& word : sampleWords) {
        std::vector<float> vec;
        if (!lookup.tryGetVector(word, vec)) {
            std::cout << "\"" << word << "\" not found in the pruned table (out of vocabulary in the raw release).\n";
            continue;
        }
        std::cout << "\"" << word << "\" -> [";
        for (size_t i = 0; i < 5 && i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i + 1 < 5 && i + 1 < vec.size()) std::cout << ", ";
        }
        std::cout << ", ... ] (" << vec.size() << " dims total)\n";
    }

    return 0;
}
