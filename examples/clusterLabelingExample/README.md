# clusterLabelingExample

Runs OpenWhiz's `owClusterLayer` through the `OpenWhiz/text/` tokenize →
TF-IDF → unsupervised-cluster → bag-of-stems-label mechanism
(`owTfIdfVectorizer` + `owClusterLabeler`) on 20 Newsgroups — a public,
category-labeled, topic-rich short-text benchmark (Hockey / Space /
Medicine / Mideast Politics / Graphics / For-Sale postings; Lang,
*NewsWeeder: Learning to Filter Netnews*, ICML 1995) — and measures cluster
purity against the known labels.

A measurement example, not a fix — `owClusterLayer.hpp` is not touched
anywhere in this example.

Run from the OpenWhiz repo root (same convention as OpenWhiz's other
`examples/*`), so `twenty_ng_sample.txt`'s relative path resolves:

```
examples/clusterLabelingExample/clusterLabelingExample
```

`twenty_ng_sample.txt` is a small (300-row, 50/class, fixed seed) stratified
subset of 20 Newsgroups' public test split (6 of its 20 categories, chosen
for topic variety), included so the example is self-contained (no download
needed to build or run it). 20 Newsgroups is scikit-learn's standard
`fetch_20newsgroups()` benchmark, with no usage restrictions.
