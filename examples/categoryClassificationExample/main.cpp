#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "OpenWhiz/text/owEmbeddingLookup.hpp"
#include "OpenWhiz/text/owSentimentPreset.hpp"
#include "OpenWhiz/text/owTextTokenizer.hpp"

// OpenWhiz/text pipeline demo: text in -> tokenize -> embed -> classify into
// one of 3+ categories, using owSentimentPreset's numClasses>=3 (softmax)
// path. Word vectors are tiny, hand-written, synthetic (4 numbers per word,
// three made-up clusters: "animal" near [1,0,0,0], "vehicle" near [0,1,0,0],
// "fruit" near [0,0,1,0]) - this demonstrates the pipeline mechanics, not
// real embedding quality. For real embeddings, prune your own vocabulary's
// vectors from a pretrained release with owWordVectorPruner.hpp.

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
    std::cout << "=== OpenWhiz/text: tokenize -> embed -> classify (3 categories) ===\n";

    std::vector<std::pair<std::string, std::vector<float>>> words = {
        {"cat", {1.0f, 0.0f, 0.0f, 0.0f}}, {"dog", {0.9f, 0.0f, 0.0f, 0.1f}},
        {"bird", {0.8f, 0.0f, 0.0f, 0.2f}}, {"pet", {0.85f, 0.0f, 0.0f, 0.15f}},
        {"car", {0.0f, 1.0f, 0.0f, 0.0f}}, {"bus", {0.0f, 0.9f, 0.0f, 0.1f}},
        {"train", {0.0f, 0.8f, 0.0f, 0.2f}}, {"drive", {0.0f, 0.85f, 0.0f, 0.15f}},
        {"apple", {0.0f, 0.0f, 1.0f, 0.0f}}, {"banana", {0.0f, 0.0f, 0.9f, 0.1f}},
        {"grape", {0.0f, 0.0f, 0.8f, 0.2f}}, {"fruit", {0.0f, 0.0f, 0.85f, 0.15f}},
    };
    writeVecFile("examples/categoryClassificationExample/words.vec", words);

    ow::owTextTokenizer tokenizer(ow::owLanguage::English);
    ow::owEmbeddingLookup lookup;
    if (!lookup.loadFromFile("examples/categoryClassificationExample/words.vec")) {
        std::cout << "failed to load words.vec\n";
        return 1;
    }

    auto embed = [&](const std::string& text) {
        return lookup.embedAverage(tokenizer.tokenize(text));
    };

    // 0=animal, 1=vehicle, 2=fruit
    std::vector<std::pair<std::string, int>> trainSentences = {
        {"I have a cat", 0}, {"My dog is happy", 0}, {"The bird sings", 0}, {"I love my pet", 0},
        {"I drive my car", 1}, {"The bus is late", 1}, {"We took the train", 1}, {"I like to drive", 1},
        {"I ate an apple", 2}, {"The banana is ripe", 2}, {"I picked a grape", 2}, {"This fruit is sweet", 2},
    };

    std::vector<std::vector<float>> trainEmb;
    std::vector<int> trainLabels;
    for (auto& ts : trainSentences) {
        trainEmb.push_back(embed(ts.first));
        trainLabels.push_back(ts.second);
    }

    ow::owSentimentPreset::Options options;
    options.hiddenSizes = {8};
    options.maxEpochs = 150;
    // All 12 synthetic examples are used for training - this is a mechanism
    // demo, not a real held-out evaluation.
    options.trainRatio = 1.0f;
    options.valRatio = 0.0f;
    options.testRatio = 0.0f;

    ow::owSentimentPreset classifier;
    std::string tempCsv = "examples/categoryClassificationExample/words.vec.train.csv";
    if (!classifier.train(trainEmb, trainLabels, 3, tempCsv, options)) {
        std::cout << "training failed\n";
        return 1;
    }

    static const char* kCategoryNames[3] = {"animal", "vehicle", "fruit"};
    std::vector<std::string> testSentences = {
        "My cat and dog are friends",
        "The train and bus were both late",
        "I ate an apple and a banana",
    };
    for (const std::string& text : testSentences) {
        std::vector<float> scores = classifier.predict(embed(text));
        int predicted = 0;
        for (int c = 1; c < 3; ++c) {
            if (scores[c] > scores[predicted]) predicted = c;
        }
        std::cout << "\"" << text << "\" -> " << kCategoryNames[predicted]
                   << " (P(animal)=" << scores[0] << " P(vehicle)=" << scores[1]
                   << " P(fruit)=" << scores[2] << ")\n";
    }

    return 0;
}
