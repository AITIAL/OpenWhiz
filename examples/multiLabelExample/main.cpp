#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "OpenWhiz/text/owEmbeddingLookup.hpp"
#include "OpenWhiz/text/owSentimentPreset.hpp"
#include "OpenWhiz/text/owTextTokenizer.hpp"

// OpenWhiz/text pipeline demo: text in -> tokenize -> embed -> classify with
// owSentimentPreset::trainMultiLabel, where a sentence can carry any number
// of the 3 labels (animal / vehicle / red) at once - unlike
// categoryClassificationExample's mutually-exclusive categories. Word
// vectors are tiny, hand-written, synthetic (4 numbers per word: one axis
// per label, plus one unused axis) - this demonstrates the pipeline
// mechanics, not real embedding quality. For real embeddings, prune your
// own vocabulary's vectors from a pretrained release with
// owWordVectorPruner.hpp.

void writeVecFile(const std::string& path, const std::vector<std::pair<std::string, std::vector<float>>>& words) {
    std::ofstream f(path);
    f << words.size() << " " << words.front().second.size() << "\n";
    for (auto& wv : words) {
        f << wv.first;
        for (float v : wv.second) f << " " << v;
        f << "\n";
    }
}

int main() {
    std::cout << "=== OpenWhiz/text: tokenize -> embed -> multi-label classify ===\n";

    std::vector<std::pair<std::string, std::vector<float>>> words = {
        {"cat", {1.0f, 0.0f, 0.0f, 0.0f}}, {"dog", {0.9f, 0.0f, 0.0f, 0.1f}},
        {"bird", {0.8f, 0.0f, 0.0f, 0.2f}},
        {"car", {0.0f, 1.0f, 0.0f, 0.0f}}, {"bus", {0.0f, 0.9f, 0.0f, 0.1f}},
        {"train", {0.0f, 0.8f, 0.0f, 0.2f}},
        {"red", {0.0f, 0.0f, 1.0f, 0.0f}}, {"crimson", {0.0f, 0.0f, 0.9f, 0.1f}},
        {"drive", {0.0f, 0.5f, 0.0f, 0.5f}}, {"have", {0.5f, 0.0f, 0.0f, 0.5f}},
    };
    writeVecFile("examples/multiLabelExample/words.vec", words);

    ow::owTextTokenizer tokenizer(ow::owLanguage::English);
    ow::owEmbeddingLookup lookup;
    if (!lookup.loadFromFile("examples/multiLabelExample/words.vec")) {
        std::cout << "failed to load words.vec\n";
        return 1;
    }

    auto embed = [&](const std::string& text) {
        return lookup.embedAverage(tokenizer.tokenize(text));
    };

    // Labels: [animal, vehicle, red] - independent, any combination allowed.
    std::vector<std::pair<std::string, std::vector<int>>> trainSentences = {
        {"I have a cat", {1, 0, 0}}, {"My dog is happy", {1, 0, 0}}, {"The bird sings", {1, 0, 0}},
        {"I drive my car", {0, 1, 0}}, {"The bus is late", {0, 1, 0}}, {"We took the train", {0, 1, 0}},
        {"I have a red cat", {1, 0, 1}}, {"My red dog is happy", {1, 0, 1}},
        {"I drive a red car", {0, 1, 1}}, {"The red bus is late", {0, 1, 1}},
        {"I have a crimson bird", {1, 0, 1}}, {"We took the red train", {0, 1, 1}},
    };

    std::vector<std::vector<float>> trainEmb;
    std::vector<std::vector<int>> trainLabels;
    for (auto& ts : trainSentences) {
        trainEmb.push_back(embed(ts.first));
        trainLabels.push_back(ts.second);
    }

    ow::owSentimentPreset::Options options;
    options.hiddenSizes = {8};
    options.maxEpochs = 400;
    // All 12 synthetic examples are used for training - this is a mechanism
    // demo, not a real held-out evaluation.
    options.trainRatio = 1.0f;
    options.valRatio = 0.0f;
    options.testRatio = 0.0f;

    ow::owSentimentPreset classifier;
    std::string tempCsv = "examples/multiLabelExample/words.vec.train.csv";
    if (!classifier.trainMultiLabel(trainEmb, trainLabels, 3, tempCsv, options)) {
        std::cout << "training failed\n";
        return 1;
    }

    static const char* kLabelNames[3] = {"animal", "vehicle", "red"};
    std::vector<std::string> testSentences = {
        "I have a red dog",
        "We took the bus",
        "My red bird and my car",
    };
    for (const std::string& text : testSentences) {
        std::vector<float> scores = classifier.predict(embed(text));
        std::cout << "\"" << text << "\" ->";
        for (int c = 0; c < 3; ++c) {
            if (scores[c] > 0.5f) std::cout << " " << kLabelNames[c];
        }
        std::cout << " (P(animal)=" << scores[0] << " P(vehicle)=" << scores[1]
                   << " P(red)=" << scores[2] << ")\n";
    }

    return 0;
}
