# sentimentClassificationExample

Runs `OpenWhiz/text`'s tokenize → stem → embed → classify pipeline through
`owSentimentPreset`'s 2-class (sigmoid) path, classifying short sentences as
positive or negative, run in English, Turkish, and French to show the same
mechanism works across languages via `owLanguage`. Also demonstrates
`owLanguageDetector` + `owMultilingualEmbeddingLookup` routing each test
sentence to its own language's embedding table.

Word vectors are tiny, hand-written, synthetic (4 numbers per word, two
made-up clusters: positive words near `[1,0,0,0]`, negative words near
`[0,0,1,0]`) — this demonstrates the pipeline mechanics, not real embedding
quality or a real sentiment lexicon. For real embeddings, prune your own
vocabulary's vectors from a pretrained release with `owWordVectorPruner.hpp`.

Run from the OpenWhiz repo root (same convention as OpenWhiz's other
`examples/*`):

```
examples/sentimentClassificationExample/sentimentClassificationExample
```
