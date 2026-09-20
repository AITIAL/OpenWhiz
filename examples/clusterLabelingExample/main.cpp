#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "OpenWhiz/text/owLanguage.hpp"
#include "OpenWhiz/text/stemmers/owStemmer.hpp"
#include "OpenWhiz/text/owClusterLabeler.hpp"
#include "OpenWhiz/text/owTfIdfVectorizer.hpp"
#include "OpenWhiz/text/owTextTokenizer.hpp"

// Measures owClusterLayer's cluster purity (via owClusterLabeler, the
// tokenize -> TF-IDF -> unsupervised-cluster -> bag-of-stems-label mechanism
// in OpenWhiz/text/) against 20 Newsgroups' known labels - a public,
// topic-rich short-text benchmark (rec.sport.hockey / sci.space / sci.med /
// talk.politics.mideast / comp.graphics / misc.forsale; Lang, "NewsWeeder:
// Learning to Filter Netnews", ICML 1995; distributed via scikit-learn's
// fetch_20newsgroups()).
//
// A measurement example, not a fix - owClusterLayer.hpp is not touched here.

static const char* kLabelNames[7] = {
    "", "Hockey", "Space", "Medicine", "MideastPolitics", "Graphics", "ForSale"
};

struct LabeledDoc {
    int trueLabel = 0;  // 1..6, see kLabelNames
    std::string title;
    std::string description;
};

std::vector<LabeledDoc> loadLabeledSample(const std::string& path) {
    std::vector<LabeledDoc> docs;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t p1 = line.find("|||");
        size_t p2 = (p1 == std::string::npos) ? std::string::npos : line.find("|||", p1 + 3);
        if (p1 == std::string::npos || p2 == std::string::npos) continue;
        LabeledDoc doc;
        doc.trueLabel = std::stoi(line.substr(0, p1));
        doc.title = line.substr(p1 + 3, p2 - (p1 + 3));
        doc.description = line.substr(p2 + 3);
        docs.push_back(doc);
    }
    return docs;
}

// owClusterLayer seeds centroids from a non-seedable std::random_device, so
// this builds an XML string with the same [-1,1] uniform distribution but a
// fixed-seed RNG, then loads it via owClusterLayer's public fromXML() right
// after construction - reproducing its default init with a fixed seed,
// without modifying owClusterLayer.hpp.
std::string buildSeededInitialCentroidsXml(int numClusters, size_t dim, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::ostringstream values;
    for (int c = 0; c < numClusters; ++c) {
        for (size_t d = 0; d < dim; ++d) {
            values << dist(rng);
            if (!(c == numClusters - 1 && d == dim - 1)) values << ' ';
        }
    }
    std::ostringstream xml;
    xml << "<InputSize>" << dim << "</InputSize>"
        << "<NumClusters>" << numClusters << "</NumClusters>"
        << "<Centroids>" << values.str() << "</Centroids>";
    return xml.str();
}

int main() {
    std::cout << "=== OpenWhiz/text: cluster labeling mechanism on 20 Newsgroups (public benchmark) ===\n\n";

    std::vector<LabeledDoc> docs = loadLabeledSample("examples/clusterLabelingExample/twenty_ng_sample.txt");
    if (docs.empty()) {
        std::cout << "Failed to load twenty_ng_sample.txt - run this from the OpenWhiz repo root.\n";
        return 1;
    }
    std::cout << "Loaded " << docs.size() << " 20 Newsgroups rows (Hockey/Space/Medicine/"
                 "MideastPolitics/Graphics/ForSale, 50 each).\n";

    ow::owTextTokenizer tokenizer(ow::owLanguage::English);
    ow::owStemmer stemmer(ow::owLanguage::English);  // no-op for English, kept for pipeline consistency

    // A short list of generic English function words - without it, "the",
    // "of", "said" etc. dominate every cluster's bag-of-stems label instead of
    // actual content words.
    static const std::unordered_set<std::string> kStopwords = {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "is", "are",
        "was", "were", "be", "been", "with", "by", "at", "as", "from", "that",
        "this", "it", "its", "has", "have", "had", "will", "after", "over",
        "new", "said", "not", "but", "or", "his", "her", "their", "than",
        "into", "up", "out", "about", "who", "which", "he", "she", "they",
        "i", "you", "we", "my", "your", "if", "so", "do", "does", "did",
        "can", "could", "would", "should", "one", "would", "also",
    };

    std::vector<std::vector<std::string>> stemLists;
    stemLists.reserve(docs.size());
    for (const LabeledDoc& doc : docs) {
        std::vector<std::string> tokens = tokenizer.tokenize(doc.title + " " + doc.description);
        std::vector<std::string> stems;
        stems.reserve(tokens.size());
        for (const std::string& tok : tokens) {
            if (kStopwords.count(tok) > 0) continue;
            std::string stemmed = stemmer.stem(tok);
            if (kStopwords.count(stemmed) > 0) continue;
            stems.push_back(stemmed);
        }
        stemLists.push_back(stems);
    }

    // Ranking vocabulary by raw document frequency alone favors generic
    // filler over discriminative terms just below the cap, so
    // maxDocFrequencyRatio excludes near-universal terms first.
    ow::owTfIdfVectorizer vectorizer;
    ow::owTfIdfVectorizer::Options vecOptions;
    vecOptions.minDocFrequency = 2;
    vecOptions.maxDocFrequencyRatio = 0.5f;
    vecOptions.maxVocabularySize = 3000;
    vecOptions.l2Normalize = true;
    std::vector<std::vector<float>> tfidfVectors = vectorizer.fitTransform(stemLists, vecOptions);
    std::cout << "TF-IDF vocabulary size: " << vectorizer.vocabulary().size() << "\n\n";

    const int numClusters = 6;  // matches this sample's 6 known classes
    const size_t dim = tfidfVectors[0].size();
    const int maxEpochs = 200;
    const float learningRate = 0.05f;

    // Reproduces owClusterLabeler::cluster()'s training loop (same layer,
    // loss, optimizer, objective) inline so a seeded centroid set can be
    // injected before training - owClusterLabeler.hpp has no seed parameter
    // and is not modified here.
    ow::owClusterLayer layer(dim, static_cast<size_t>(numClusters));
    layer.fromXML(buildSeededInitialCentroidsXml(numClusters, dim, /*seed=*/42));

    ow::owADAMOptimizer optimizer(learningRate);
    layer.setOptimizer(&optimizer);
    ow::owMeanSquaredErrorLoss loss;

    ow::owTensor<float, 2> input(tfidfVectors.size(), dim);
    for (size_t i = 0; i < tfidfVectors.size(); ++i) {
        for (size_t d = 0; d < dim; ++d) input(i, d) = tfidfVectors[i][d];
    }
    ow::owTensor<float, 2> target(tfidfVectors.size(), static_cast<size_t>(numClusters));
    target.setZero();

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
        ow::owTensor<float, 2> output = layer.forward(input);
        ow::owTensor<float, 2> grad = loss.gradient(output, target);
        layer.backward(grad);
        layer.train();
    }

    ow::owTensor<float, 2> finalDistances = layer.forward(input);
    std::vector<int> assignments(tfidfVectors.size());
    for (size_t i = 0; i < tfidfVectors.size(); ++i) {
        int best = 0;
        float bestDist = finalDistances(i, 0);
        for (int c = 1; c < numClusters; ++c) {
            if (finalDistances(i, c) < bestDist) { bestDist = finalDistances(i, c); best = c; }
        }
        assignments[i] = best;
    }

    // Bag-of-stems labeling: names each cluster from its most frequent
    // stems. Replicated directly here since owClusterLabeler::cluster()
    // recomputes clustering internally rather than accepting fixed
    // assignments.
    const int numTrueLabels = 6;

    int totalMajorityCorrect = 0;
    for (int c = 0; c < numClusters; ++c) {
        std::unordered_map<std::string, int> stemFreq;
        std::vector<int> counts(numTrueLabels + 1, 0);
        int size = 0;
        for (size_t i = 0; i < assignments.size(); ++i) {
            if (assignments[i] != c) continue;
            ++size;
            counts[docs[i].trueLabel]++;
            for (const std::string& stem : stemLists[i]) stemFreq[stem]++;
        }
        std::vector<std::pair<std::string, int>> topStems(stemFreq.begin(), stemFreq.end());
        std::sort(topStems.begin(), topStems.end(),
                  [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                      return a.second > b.second;
                  });
        if (topStems.size() > 8) topStems.resize(8);

        int majorityLabel = 1;
        for (int lbl = 2; lbl <= numTrueLabels; ++lbl) {
            if (counts[lbl] > counts[majorityLabel]) majorityLabel = lbl;
        }
        totalMajorityCorrect += counts[majorityLabel];

        std::cout << "=== Cluster " << c << " (n=" << size << ") ===\n";
        std::cout << "Top stems:";
        for (const auto& stemFreq2 : topStems) std::cout << " " << stemFreq2.first << "(" << stemFreq2.second << ")";
        std::cout << "\n";
        std::cout << "True-label breakdown:";
        for (int lbl = 1; lbl <= numTrueLabels; ++lbl) {
            std::cout << " " << kLabelNames[lbl] << "=" << counts[lbl];
        }
        std::cout << " (majority=" << kLabelNames[majorityLabel] << ")\n\n";
    }

    float purity = 100.0f * totalMajorityCorrect / docs.size();
    std::cout << "Purity: " << purity << "% (chance baseline for 6 balanced classes: ~16.7%)\n";
    std::cout << "(" << totalMajorityCorrect << "/" << docs.size() << " documents in their cluster's majority class)\n";

    return 0;
}
