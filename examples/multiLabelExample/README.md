# multiLabelExample

Runs `OpenWhiz/text`'s tokenize → embed → classify pipeline through
`owSentimentPreset::trainMultiLabel`, where a sentence can carry any
number of 3 independent labels (animal / vehicle / red) at once — unlike
`categoryClassificationExample`'s mutually-exclusive categories.

Word vectors are tiny, hand-written, synthetic (4 numbers per word) — this
demonstrates the pipeline mechanics, not real embedding quality. For real
embeddings, prune your own vocabulary's vectors from a pretrained release
with `owWordVectorPruner.hpp`.

Run from the OpenWhiz repo root (same convention as OpenWhiz's other
`examples/*`):

```
examples/multiLabelExample/multiLabelExample
```
