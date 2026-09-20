#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>

#include "OpenWhiz/text/owLanguage.hpp"
#include "OpenWhiz/text/owLanguageDetector.hpp"
#include "OpenWhiz/text/owMultilingualEmbeddingLookup.hpp"
#include "OpenWhiz/text/stemmers/owStemmer.hpp"
#include "OpenWhiz/text/owEmbeddingLookup.hpp"
#include "OpenWhiz/text/owSentimentPreset.hpp"
#include "OpenWhiz/text/owTextTokenizer.hpp"

// OpenWhiz/text pipeline demo: text in -> tokenize -> stem -> embed -> classify
// (positive/negative sentiment), run in English, Turkish, and French to show
// the SAME mechanism works across languages via owLanguage. The word vectors
// below are tiny, hand-written, synthetic (4 numbers per word, two made-up
// clusters: positive words near [1,0,0,0], negative words near [0,0,1,0]) -
// this demonstrates the pipeline mechanics, not real embedding quality or
// real sentiment lexicon data. For real embeddings, prune your own
// vocabulary's vectors from a pretrained release with owWordVectorPruner.hpp -
// real vocabulary/vector data is deliberately never shipped in the library
// itself (see that header's and OpenWhiz/text/README.md's notes on why).
//
// Each language's sentences intentionally use only exact vocabulary word
// forms (owStemmer still runs on every token; for these particular words it
// has nothing to reduce).

struct LangSample {
    ow::owLanguage language;
    std::string name;
    std::string vecFile;
    std::vector<std::pair<std::string, std::vector<float>>> words;
    std::vector<std::pair<std::string, int>> trainSentences; // text, label (0=negative, 1=positive)
    std::vector<std::string> testSentences;
};

void writeVecFile(const std::string& path, const std::vector<std::pair<std::string, std::vector<float>>>& words) {
    std::ofstream f(path);
    f << words.size() << " " << words.front().second.size() << "\n";
    for (auto& wv : words) {
        f << wv.first;
        for (float v : wv.second) f << " " << v;
        f << "\n";
    }
}

void runLanguageDemo(const LangSample& sample) {
    std::cout << "\n=== " << sample.name << " ===\n";
    writeVecFile(sample.vecFile, sample.words);

    ow::owTextTokenizer tokenizer(sample.language);
    ow::owStemmer stemmer(sample.language);
    ow::owEmbeddingLookup lookup;
    if (!lookup.loadFromFile(sample.vecFile)) {
        std::cout << "failed to load " << sample.vecFile << "\n";
        return;
    }

    auto embed = [&](const std::string& text) {
        std::vector<std::string> tokens = tokenizer.tokenize(text);
        std::vector<std::string> stemmed;
        for (auto& t : tokens) stemmed.push_back(stemmer.stem(t));
        return lookup.embedAverage(stemmed);
    };

    std::vector<std::vector<float>> trainEmb;
    std::vector<int> trainLabels;
    for (auto& ts : sample.trainSentences) {
        trainEmb.push_back(embed(ts.first));
        trainLabels.push_back(ts.second);
    }

    ow::owSentimentPreset::Options options;
    options.hiddenSizes = {4};
    options.maxEpochs = 300;
    // All 8 synthetic examples are used for training - this is a mechanism demo,
    // not a real held-out evaluation.
    options.trainRatio = 1.0f;
    options.valRatio = 0.0f;
    options.testRatio = 0.0f;

    ow::owSentimentPreset classifier;
    std::string tempCsv = sample.vecFile + ".train.csv";
    if (!classifier.train(trainEmb, trainLabels, 2, tempCsv, options)) {
        std::cout << "training failed\n";
        return;
    }

    for (auto& text : sample.testSentences) {
        auto scores = classifier.predict(embed(text));
        std::string predicted = scores[1] > scores[0] ? "positive" : "negative";
        std::cout << "\"" << text << "\" -> " << predicted
                   << " (P(negative)=" << scores[0] << " P(positive)=" << scores[1] << ")\n";
    }
}

int main() {
    std::cout << "=== OpenWhiz/text: tokenize -> stem -> embed -> classify (positive/negative, EN/TR/FR) ===\n";

    LangSample english{
        ow::owLanguage::English, "English", "examples/sentimentClassificationExample/en.vec",
        {
            {"love", {1.0f, 0.0f, 0.0f, 0.0f}}, {"amazing", {0.9f, 0.1f, 0.0f, 0.0f}},
            {"wonderful", {0.8f, 0.2f, 0.0f, 0.0f}}, {"great", {0.85f, 0.15f, 0.0f, 0.0f}},
            {"hate", {0.0f, 0.0f, 1.0f, 0.0f}}, {"terrible", {0.0f, 0.0f, 0.9f, 0.1f}},
            {"awful", {0.0f, 0.0f, 0.8f, 0.2f}}, {"horrible", {0.0f, 0.0f, 0.85f, 0.15f}},
        },
        {
            {"I love this", 1}, {"This is amazing", 1}, {"What a wonderful day", 1}, {"This is great", 1},
            {"I hate this", 0}, {"This is terrible", 0}, {"What an awful day", 0}, {"This is horrible", 0},
        },
        {"This is truly amazing and wonderful", "This is terrible and awful"}
    };

    LangSample turkish{
        ow::owLanguage::Turkish, "Turkish", "examples/sentimentClassificationExample/tr.vec",
        {
            {"harika", {1.0f, 0.0f, 0.0f, 0.0f}}, {"güzel", {0.9f, 0.1f, 0.0f, 0.0f}},
            {"mükemmel", {0.8f, 0.2f, 0.0f, 0.0f}}, {"keyifli", {0.85f, 0.15f, 0.0f, 0.0f}},
            {"berbat", {0.0f, 0.0f, 1.0f, 0.0f}}, {"kötü", {0.0f, 0.0f, 0.9f, 0.1f}},
            {"korkunç", {0.0f, 0.0f, 0.8f, 0.2f}}, {"rezil", {0.0f, 0.0f, 0.85f, 0.15f}},
        },
        {
            {"Bu çok harika", 1}, {"Hava çok güzel", 1}, {"Sonuç mükemmel", 1}, {"Film keyifli", 1},
            {"Bu çok berbat", 0}, {"Hava çok kötü", 0}, {"Sonuç korkunç", 0}, {"Film rezil", 0},
        },
        {"Bu film harika ve güzel", "Bu film berbat ve korkunç"}
    };

    LangSample french{
        ow::owLanguage::French, "French", "examples/sentimentClassificationExample/fr.vec",
        {
            {"incroyable", {1.0f, 0.0f, 0.0f, 0.0f}}, {"génial", {0.9f, 0.1f, 0.0f, 0.0f}},
            {"merveilleux", {0.8f, 0.2f, 0.0f, 0.0f}}, {"magnifique", {0.85f, 0.15f, 0.0f, 0.0f}},
            {"horrible", {0.0f, 0.0f, 1.0f, 0.0f}}, {"terrible", {0.0f, 0.0f, 0.9f, 0.1f}},
            {"désastreux", {0.0f, 0.0f, 0.8f, 0.2f}}, {"épouvantable", {0.0f, 0.0f, 0.85f, 0.15f}},
        },
        {
            {"C'est incroyable", 1}, {"C'est génial", 1}, {"C'est merveilleux", 1}, {"C'est magnifique", 1},
            {"C'est horrible", 0}, {"C'est terrible", 0}, {"C'est désastreux", 0}, {"C'est épouvantable", 0},
        },
        {"C'est vraiment génial et merveilleux", "C'est désastreux et épouvantable"}
    };

    runLanguageDemo(english);
    runLanguageDemo(turkish);
    runLanguageDemo(french);

    // Detects each test sentence's language and shows it routing to that
    // language's own embedding table via owMultilingualEmbeddingLookup.
    std::cout << "\n=== owLanguageDetector + owMultilingualEmbeddingLookup ===\n";
    ow::owMultilingualEmbeddingLookup multiLookup;
    multiLookup.loadForLanguage(ow::owLanguage::English, english.vecFile);
    multiLookup.loadForLanguage(ow::owLanguage::Turkish, turkish.vecFile);
    multiLookup.loadForLanguage(ow::owLanguage::French, french.vecFile);
    for (const LangSample* sample : {&english, &turkish, &french}) {
        for (const std::string& text : sample->testSentences) {
            ow::owLanguage detected = ow::owLanguageDetector::detect(text);
            const char* name = detected == ow::owLanguage::English ? "English"
                              : detected == ow::owLanguage::Turkish ? "Turkish" : "French";
            std::cout << "\"" << text << "\" -> detected=" << name
                       << ", routed to " << name << "'s embedding table\n";
        }
    }

    return 0;
}
