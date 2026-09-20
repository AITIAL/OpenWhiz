#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ow {

// Word -> (positive, negative) polarity score lookup, loaded from a public
// sentiment lexicon, used to auto-label text without a trained classifier
// (bag-of-words scoring: average the matched words' scores). Supporting a
// new language only requires providing that language's lexicon file.
//
// Four loaders for four common lexicon distribution formats:
//  - loadSentiTurkNetXml(): StarlangSoftware/TurkishSentiNet's
//    turkish_sentiliteralnet.xml - flat <WORD><NAME>/<PSCORE>/<NSCORE> records,
//    one word per synset-independent literal, parsed directly (no XML library
//    needed, the format is flat/simple enough for direct string scanning).
//  - loadSentiWordNetText(): aesuli/SentiWordNet's SentiWordNet_3.0.0.txt -
//    tab-separated POS/ID/PosScore/NegScore/SynsetTerms/Gloss, one row per
//    WordNet synset; a word can appear in many synsets with different scores,
//    so this averages a word's score across every synset it appears in.
//  - loadOpenerLmf(): OpeNER-LMF XML (<LexicalEntry><Lemma writtenForm="..."/>
//    <Sense><Sentiment polarity="..."/><Confidence score="..."/></Sense>
//    </LexicalEntry>) - a word can have multiple Sense entries, averaged the
//    same way loadSentiWordNetText() averages across synsets. Parsed by direct
//    string scanning, no XML library needed.
//  - loadPlainTsv(): generic "word\tposScore\tnegScore" format for any source
//    that isn't one of the above three.
class owSentimentLexicon {
public:
    bool loadSentiTurkNetXml(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) return false;

        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        size_t pos = 0;
        while ((pos = content.find("<WORD>", pos)) != std::string::npos) {
            size_t end = content.find("</WORD>", pos);
            if (end == std::string::npos) break;
            std::string entry = content.substr(pos, end - pos);

            std::string name = extractTag(entry, "NAME");
            std::string pscore = extractTag(entry, "PSCORE");
            std::string nscore = extractTag(entry, "NSCORE");
            if (!name.empty() && !pscore.empty() && !nscore.empty()) {
                try {
                    m_scores[name] = {std::stof(pscore), std::stof(nscore)};
                } catch (...) {
                    // malformed numeric field - skip this entry, keep going
                }
            }
            pos = end + 7;
        }
        return true;
    }

    bool loadSentiWordNetText(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) return false;

        std::unordered_map<std::string, std::pair<float, float>> sums;
        std::unordered_map<std::string, int> counts;

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            std::string posTag, id, posScoreStr, negScoreStr, synsetTerms, gloss;
            if (!std::getline(ls, posTag, '\t')) continue;
            if (!std::getline(ls, id, '\t')) continue;
            if (!std::getline(ls, posScoreStr, '\t')) continue;
            if (!std::getline(ls, negScoreStr, '\t')) continue;
            if (!std::getline(ls, synsetTerms, '\t')) continue;

            float posScore = 0.0f, negScore = 0.0f;
            try {
                posScore = std::stof(posScoreStr);
                negScore = std::stof(negScoreStr);
            } catch (...) {
                continue;
            }

            std::istringstream terms(synsetTerms);
            std::string term;
            while (terms >> term) {
                size_t hash = term.find('#');
                std::string word = hash == std::string::npos ? term : term.substr(0, hash);
                auto& sum = sums[word];
                sum.first += posScore;
                sum.second += negScore;
                counts[word] += 1;
            }
        }

        for (auto& kv : sums) {
            int n = counts[kv.first];
            if (n > 0) m_scores[kv.first] = {kv.second.first / n, kv.second.second / n};
        }
        return true;
    }

    bool loadOpenerLmf(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) return false;

        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::unordered_map<std::string, std::pair<float, float>> sums;
        std::unordered_map<std::string, int> counts;

        size_t pos = 0;
        while ((pos = content.find("<LexicalEntry", pos)) != std::string::npos) {
            size_t end = content.find("</LexicalEntry>", pos);
            if (end == std::string::npos) break;
            std::string entry = content.substr(pos, end - pos);

            std::string word = extractAttr(entry, "Lemma", "writtenForm");
            if (!word.empty()) {
                size_t sensePos = 0;
                while ((sensePos = entry.find("<Sense", sensePos)) != std::string::npos) {
                    size_t senseEnd = entry.find("</Sense>", sensePos);
                    if (senseEnd == std::string::npos) break;
                    std::string sense = entry.substr(sensePos, senseEnd - sensePos);

                    if (sense.find("<Sentiment") != std::string::npos) {
                        std::string polarity = extractAttr(sense, "Sentiment", "polarity");
                        if (polarity.empty()) polarity = "neutral";
                        std::string scoreStr = extractAttr(sense, "Confidence", "score");
                        float score = 0.0f;
                        if (!scoreStr.empty()) {
                            try { score = std::stof(scoreStr); } catch (...) {}
                        }
                        auto& sum = sums[word];
                        sum.first += polarity == "positive" ? score : 0.0f;
                        sum.second += polarity == "negative" ? score : 0.0f;
                        counts[word] += 1;
                    }
                    sensePos = senseEnd + 8;
                }
            }
            pos = end + 15;
        }

        for (auto& kv : sums) {
            int n = counts[kv.first];
            if (n > 0) m_scores[kv.first] = {kv.second.first / n, kv.second.second / n};
        }
        return true;
    }

    bool loadPlainTsv(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) return false;

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream ls(line);
            std::string word, posStr, negStr;
            if (!std::getline(ls, word, '\t')) continue;
            if (!std::getline(ls, posStr, '\t')) continue;
            if (!std::getline(ls, negStr, '\t')) continue;
            try {
                m_scores[word] = {std::stof(posStr), std::stof(negStr)};
            } catch (...) {
                continue;
            }
        }
        return true;
    }

    bool tryGetScore(const std::string& word, float& posOut, float& negOut) const {
        auto it = m_scores.find(word);
        if (it == m_scores.end()) return false;
        posOut = it->second.first;
        negOut = it->second.second;
        return true;
    }

    size_t getVocabularySize() const { return m_scores.size(); }

    struct TextScore {
        float pos = 0.0f;
        float neg = 0.0f;
        int matchedWords = 0;
    };

    // Bag-of-words average over the words in `tokens` that are present in the
    // lexicon; words not found are silently skipped (same convention as
    // owEmbeddingLookup::embedAverage()).
    TextScore scoreTokens(const std::vector<std::string>& tokens) const {
        TextScore result;
        for (const std::string& token : tokens) {
            float pos = 0.0f, neg = 0.0f;
            if (!tryGetScore(token, pos, neg)) continue;
            result.pos += pos;
            result.neg += neg;
            ++result.matchedWords;
        }
        if (result.matchedWords > 0) {
            result.pos /= result.matchedWords;
            result.neg /= result.matchedWords;
        }
        return result;
    }

private:
    static std::string extractTag(const std::string& entry, const std::string& tag) {
        std::string open = "<" + tag + ">";
        std::string close = "</" + tag + ">";
        size_t start = entry.find(open);
        if (start == std::string::npos) return "";
        start += open.size();
        size_t end = entry.find(close, start);
        if (end == std::string::npos) return "";
        return entry.substr(start, end - start);
    }

    // Returns the value of attr="..." on the first <tag ...> found in entry.
    static std::string extractAttr(const std::string& entry, const std::string& tag, const std::string& attr) {
        size_t tagPos = entry.find("<" + tag);
        if (tagPos == std::string::npos) return "";
        size_t tagEnd = entry.find('>', tagPos);
        if (tagEnd == std::string::npos) return "";
        std::string tagContent = entry.substr(tagPos, tagEnd - tagPos);

        std::string attrKey = attr + "=\"";
        size_t attrPos = tagContent.find(attrKey);
        if (attrPos == std::string::npos) return "";
        attrPos += attrKey.size();
        size_t attrEnd = tagContent.find('"', attrPos);
        if (attrEnd == std::string::npos) return "";
        return tagContent.substr(attrPos, attrEnd - attrPos);
    }

    std::unordered_map<std::string, std::pair<float, float>> m_scores;
};

} // namespace ow
