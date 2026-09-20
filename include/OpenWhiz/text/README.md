# OpenWhiz/text

Text-processing extension for OpenWhiz — tokenization, stemming, embedding lookup,
and a sentiment classification preset built on OpenWhiz's existing layers/losses.
Lives inside OpenWhiz's own include tree (`OpenWhiz/text/...`), so it reads as a
natural extension of OpenWhiz from the caller's side and shares the `ow::`
namespace (`ow::owTextTokenizer`, `ow::owSentimentPreset`, ...).

Not wired into `engine/CMakeLists.txt` — `libs/OpenWhiz/include` is already on every
GlistEngine project's include path, so `#include "OpenWhiz/text/..."` just works with
no extra CMake changes; projects that never `#include` these headers pay no cost.

Vocabulary extraction and vector pruning (`owVocabularyExtractor.hpp`/
`owWordVectorPruner.hpp` below) are header-only C++, not external scripts —
`#include` them directly from your own project's one-off data-prep code.
Language data itself (pretrained vectors, a project's own vocabulary, labeled
datasets) is **not** shipped here — each project prunes/produces its own this
way and keeps the result in its own data directory.

Four runnable samples under `examples/*` (same layout/build convention as
OpenWhiz's other examples) each show one capability end-to-end:
- `sentimentClassificationExample/` — tokenize -> stem -> embed -> classify
  (2-class, positive/negative), in English, Turkish, and French, using a
  tiny hand-written synthetic word-vector table (not real embeddings — see
  that example's own comments).
- `categoryClassificationExample/` — the same pipeline through
  `owSentimentPreset`'s 3+-class (softmax) path, classifying into one of
  three mutually-exclusive categories.
- `multiLabelExample/` — the same pipeline through `trainMultiLabel()`,
  where a sentence can carry any number of independent labels at once.
- `clusterLabelingExample/` — tokenize -> TF-IDF -> unsupervised-cluster ->
  bag-of-stems-label on a public, category-labeled dataset (20 Newsgroups),
  printing a purity metric against the known labels — see that example's
  README for the measurement and what it covers.

## Headers

- `owLanguage.hpp` — `enum class owLanguage { English, Turkish, French }`, shared by
  `owTextTokenizer` and `owStemmer`. Default is `English` everywhere — callers that
  need Turkish/French tokenization or stemming must pass the language explicitly
  (don't rely on the default for a non-English corpus).
- `owTextTokenizer.hpp` — UTF-8 aware word-level tokenizer, constructed
  with an `owLanguage` (default `English`). Splits on anything that isn't a
  letter/digit; language controls which accented letters count as word characters
  and how they lowercase — English is plain ASCII, Turkish adds Ç Ğ İ Ö Ş Ü with
  the Turkish-specific ASCII `I` -> dotless `ı` / `İ` -> ASCII `i` mapping (generic
  ASCII case folding gets this wrong for Turkish), French adds À Â Ä Ç É È Ê Ë Î Ï
  Ô Ö Ù Û Ü Ÿ Œ Æ with standard case folding (no special-case pairs needed).
- `owLanguageDetector.hpp` — heuristic per-text language detector (Turkish/
  English/French) for short, mixed-language content like list titles —
  not a general-purpose language classifier. Produces the same `owLanguage`
  value consumed by `owTextTokenizer`, `owStemmer`, and
  `owMultilingualEmbeddingLookup`, so one detection result can drive all
  three consistently instead of each caller guessing independently.
- `stemmers/owStemmer.hpp` — language-dispatching facade over the per-language stemmers
  under `stemmers/`; construct with an `owLanguage` and call `stem()`.
- `stemmers/owTurkishStemmer.hpp` — rule-based Turkish suffix stripping (kök
  bulma), used directly by `owStemmer` for `owLanguage::Turkish`. No dictionary, no
  morphological analyzer, no external dependency. Heuristic longest-suffix-first
  stripping across a few passes (see header comment for what it does and doesn't
  model).
- `stemmers/owFrenchStemmer.hpp` — rule-based French suffix stripping (plural -s/-x,
  feminine -e, a handful of verb endings), used by `owStemmer` for
  `owLanguage::French`. Deliberately simpler than the Turkish table — see header
  comment for what it does and doesn't model.
- `stemmers/owEnglishStemmer.hpp` — no-op passthrough used by `owStemmer` for
  `owLanguage::English`; kept as its own class (same shape as the other two) so the
  dispatch is uniform and there's a single place to add real stemming for English
  if a use case ever needs it.
  Per-language stemmers live under `stemmers/`; `owTextTokenizer.hpp` is a single
  file directly under `text/` since there's only one (language-parameterized)
  tokenizer, not one per language.
- `owEmbeddingLookup.hpp` — loads a pretrained fastText-format word vector table
  (word + N floats per line) into an in-memory lookup, plus a mean-pooling
  `embedAverage()` helper for turning a token list into one fixed-size vector.
  Expects the table already pruned to a project's real vocabulary — see
  `owWordVectorPruner.hpp` — not the full multi-GB release.
- `owMultilingualEmbeddingLookup.hpp` — language-dispatching facade over
  `owEmbeddingLookup`, holding one word-vector table per `owLanguage` and
  routing `embedAverage()`/`tryGetVector()` to the right one (same dispatch
  shape as `owStemmer`). Exists because different languages' word-vector
  tables are separately trained, unaligned vector spaces — a Turkish
  embedding and an English embedding are not comparable by cosine similarity
  or any other distance, even at the same dimension — so this class only
  exposes each table through its own `owLanguage`, with no "compare across
  languages" method, on purpose. Pair with `owLanguageDetector` to pick which
  language a given piece of text should be embedded (and tokenized/stemmed)
  as.
- `owWordVectorPruner.hpp` — prunes a full pretrained word-vector table (fastText/
  word2vec-text format, or classic word2vec.c binary format) down to a project's
  actual vocabulary, producing a small table `owEmbeddingLookup` can load. A
  library header, not an external script — call it from your own one-off
  data-prep code.
- `owVocabularyExtractor.hpp` — builds that vocabulary from a raw text corpus
  using `owTextTokenizer`, so it matches the tokenization rules used at lookup
  time.
- `owSentimentPreset.hpp` — sentiment classification head over embedding vectors
  (e.g. from `owEmbeddingLookup::embedAverage()`). 2-class -> sigmoid output +
  `owBinaryCrossEntropyLoss`; 3+ class -> `owProbabilityLayer` (softmax) +
  `owCategoricalCrossEntropyLoss` — same pattern as OpenWhiz's own
  `classificationExample`. `trainMultiLabel()` covers the independent-labels case
  instead (a row can carry any number of a fixed label set at once, e.g. both
  "animal" and "red") — a Sigmoid output sized to the label count +
  `owBinaryCrossEntropyLoss`/`owWeightedBinaryCrossEntropyLoss`, no softmax
  normalization across labels. `owDataset` has no public API to inject a numeric
  matrix directly — only `loadFromCSV()` populates it — so `train()`/
  `trainMultiLabel()` here write embeddings+labels to a temp CSV and load them
  that way; this is how every OpenWhiz example actually feeds data in, not a
  workaround. `Options::trainRatio/valRatio/testRatio` expose `owDataset`'s split
  ratios (default 0.6/0.2/0.2) — on small datasets this can starve the training
  set (e.g. 8 samples -> only ~4 actually trained on), so tests/small experiments
  should widen `trainRatio`.
- `owSentimentLexicon.hpp` — word -> (positive, negative) polarity score lookup
  for auto-labeling text without a trained classifier (bag-of-words scoring via
  `scoreTokens()`). Four loaders for four common public lexicon distribution
  formats: `loadSentiTurkNetXml()` (StarlangSoftware/TurkishSentiNet's flat
  `<WORD><NAME>/<PSCORE>/<NSCORE>` XML), `loadSentiWordNetText()` (aesuli/
  SentiWordNet's tab-separated synset format, averaged per word across synsets),
  `loadOpenerLmf()` (OpeNER-LMF XML, `<LexicalEntry><Lemma>/<Sense><Sentiment>`,
  averaged per word across senses), and `loadPlainTsv()` (generic
  `word\tposScore\tnegScore` for any other source). None of these formats' data
  is shipped here — point each loader at your own downloaded lexicon file.
- `owClusterLabeler.hpp` — unsupervised clustering over a set of embedding
  vectors (e.g. from `owEmbeddingLookup::embedAverage()`), plus a proposed
  human-readable label per cluster from its most frequent stemmed tokens
  (bag-of-stems frequency count). Reuses `owClusterLayer` directly (centroid
  distances, trained by pulling them toward the data with MSE-to-zero +
  `owADAMOptimizer` — same objective as `examples/clusterExample`); no
  `owDataset`/CSV round-trip needed since `owClusterLayer` takes a raw
  `owTensor` input. Deliberately a stateless, one-shot static method rather
  than a constructed/held object like OpenWhiz's own layers - there's no
  trained model to keep around afterward (unlike `owSentimentPreset`,
  which does hold a trained network for repeated `predict()` calls).
- `owTfIdfVectorizer.hpp` — TF-IDF over already-tokenized (e.g. stemmed)
  documents: fits a vocabulary + smoothed IDF from a corpus, returns one dense
  L2-normalized TF-IDF vector per document. A bag-of-words alternative to
  `owEmbeddingLookup::embedAverage()` for when averaging pretrained vectors
  dilutes topic signal too much on short documents — output plugs into
  `owClusterLabeler::cluster()` the same way `embedAverage()` output does,
  since that function doesn't care what the vectors mean. `Options::
  maxVocabularySize` caps the vocabulary to the N highest-document-frequency
  terms (after `minDocFrequency`) — on a lexically rich corpus this can favor
  near-universal filler over the actually-discriminative terms sitting just
  below the cap, so also set `maxDocFrequencyRatio` (e.g. 0.5) to drop terms
  above that document-frequency ceiling first.

Stopword filtering before `embedAverage()` is project-specific (which filler words
hurt category signal depends on the corpus) and so is NOT shipped here — write a
small stopword list tuned against your own corpus and filter tokens with it before
calling `embedAverage()`.

## Choosing a training pipeline

`writeTrainingCSV()` (inside `owSentimentPreset::train()`) writes embedding floats
with `std::fixed` at high precision on purpose: `owDataset`'s numeric-column
detector does not recognize scientific notation (`1.23e-05`), and a single such
value silently turns an entire column into a text/category column instead of a
numeric one. Don't remove the `std::fixed`/`setprecision` formatting when touching
that function.
