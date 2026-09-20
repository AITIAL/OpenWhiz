# categoryClassificationExample

Runs `OpenWhiz/text`'s tokenize → embed → classify pipeline through
`owSentimentPreset`'s `numClasses>=3` (softmax) path, classifying short
sentences into one of three categories (animal / vehicle / fruit) instead
of the binary sentiment case shown in `sentimentClassificationExample`.

Word vectors are tiny, hand-written, synthetic (4 numbers per word, three
made-up clusters) — this demonstrates the pipeline mechanics, not real
embedding quality. For real embeddings, prune your own vocabulary's
vectors from a pretrained release with `owWordVectorPruner.hpp`.

Run from the OpenWhiz repo root (same convention as OpenWhiz's other
`examples/*`):

```
examples/categoryClassificationExample/categoryClassificationExample
```
