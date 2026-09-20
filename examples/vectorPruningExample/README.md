# vectorPruningExample

Runs `OpenWhiz/text`'s vocabulary → pruning → lookup pipeline against a
**real** pretrained word-vector release, instead of the tiny hand-written
synthetic tables used by `sentimentClassificationExample`/
`categoryClassificationExample`/`multiLabelExample`: extracts a vocabulary
from a small sample corpus with `owVocabularyExtractor`, prunes a full
multi-GB pretrained release down to just those words with
`owWordVectorPruner`, then loads the small pruned table with
`owEmbeddingLookup` and prints a few real vectors.

## Getting the raw vector file

**The raw pretrained file is not included in this repository** — it's
several GB, unlicensed for redistribution here, and each project should
provide its own. Download `GoogleNews-vectors-negative300.bin` (the classic
word2vec-binary-format release trained on ~100B words of Google News text)
from the community mirror
[mmihaltz/word2vec-GoogleNews-vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)
and place it at:

```
examples/vectorPruningExample/GoogleNews-vectors-negative300.bin
```

Any other word2vec-binary-format release works too, as long as it's placed
at that exact path — `owWordVectorPruner::pruneWord2VecBinaryFormat()` reads
the classic word2vec.c binary format (ASCII `<count> <dim>` header, then
repeated `<word><space><dim x float32>` entries), which is what several
major pretrained releases (including GoogleNews) are distributed in.

If the file isn't present, this example prints that message and exits
cleanly (does not crash) instead of failing on a missing file.

## Run

Run from the OpenWhiz repo root (same convention as OpenWhiz's other
`examples/*`):

```
examples/vectorPruningExample/vectorPruningExample
```
