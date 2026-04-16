# OpenWhiz ⚡

**OpenWhiz** is a high-performance, zero-dependency and header-only C++ **Deep Learning Framework for Financial and Industrial Systems**. Designed for seamless integration into Desktop, Mobile, Web (via WASM), and Edge/IoT devices, it focuses on mathematical precision, execution speed, and high-level architectural flexibility. Developed by **AITIAL Paris**.

---

## 🚀 Key Features

*   **Zero-Dependency:** No external libraries required (no BLAS, no Protobuf, no Eigen). Just include and compile.
*   **Header-Only:** Easy to integrate into any build system (CMake, Make, MSVC).
*   **Advanced Temporal Engine:** Built-in sliding window strategies and caching mechanisms optimized for financial and industrial time-series.
*   **CPU Optimized:** Leveraging SIMD instructions (AVX-512, AVX2, SSE, ARM NEON) and multi-core processing via OpenMP.
*   **C++14 Compliant:** Modern, safe, and compatible with established embedded and industrial toolchains.
*   **Lightweight & Fast:** Minimal memory footprint and deterministic performance, making it ideal for real-time controllers and high-frequency modeling.

---

## 💎 Signature Feature: The Temporal Cache System

OpenWhiz provides a unique, high-performance architecture for handling time-series data:

### 3-Tier Sliding Window Strategy
*   **Dataset-Level (Static):** Use `dataset->prepareForecastData()` to pre-transform data for rapid prototyping.
*   **View-Layer (Hierarchical):** Use `owSlidingWindowViewLayer` to extract multiple temporal resolutions (e.g., short-term vs long-term trends) from a single master dataset without redundant copies.
*   **In-Model Pipeline (Surgical):** Use `owSlidingWindowLayer` + `owCacheLayer`. Keep your dataset pure; the model creates windows on-the-fly and caches them for future epochs.

### The Power of `owCacheLayer`
By placing a `owCacheLayer` after non-trainable preprocessing layers:
*   **O(1) Playback:** Expensive preprocessing (Normalization, Windowing) is performed once during the first epoch and replayed directly from cache thereafter.
*   **Temporal Integrity:** Shuffling is applied to the cached windows, preserving internal time-sequences while randomizing sample order for better generalization.

---

## 🎭 Signature Feature: Expert-Ensemble Architecture

OpenWhiz enables the creation of **Independent Experts** using the `owConcatenateLayer`. This allows the model to analyze data through multiple parallel "brains" simultaneously.

*   **Multi-Perspective Analysis:** You can build a branch for weekly forecasting and another for monthly analysis within the same model.
*   **Parallel Processing:** Unlimited branches can be added, each containing its own sequence of layers (LSTMs, Linear, etc.).
*   **Joint Decision Making:** The results from all experts are concatenated and fed into a final set of layers that learn to weigh each expert's opinion based on the current data context.

---

## 🏗️ Architecture Overview

OpenWhiz is structured into several modular components:

*   **Core:** High-performance `owTensor` engine and `owNeuralNetwork` manager.
*   **Layers:** From standard `Linear` and `LSTMLayer` to specialized `owSlidingWindowLayer` and `owCacheLayer`.
*   **Optimizers:** First-order (SGD, Adam) and second-order (L-BFGS, Conjugate Gradient) methods.
*   **Data:** `owDataset` for CSV handling, automated normalization, and statistical profiling.
*   **Activations & Losses:** A wide range of non-linearities (ReLU, Tanh, Sigmoid) and loss functions (MSE, Huber, Cross-Entropy).

---

## 🧩 Project Types & Deep Learning Paradigms

OpenWhiz simplifies network construction through high-level project types:

### 🎯 Supervised Learning
*   **APPROXIMATION:** Continuous function fitting (Regression) for industrial modeling and physical simulations.
*   **FORECASTING:** Time-series prediction for financial markets, demand planning, and predictive maintenance.
*   **CLASSIFICATION:** Categorical prediction (Multi-class/Binary) for decision-making and pattern recognition.

### 🔍 Unsupervised Learning
*   **CLUSTERING:** Grouping similar data points using projection and distance metrics (Latent Space Analysis).
*   **ANOMALY_DETECTION:** Identifying outliers in data streams (Statistical Z-Score & Projection).

### 🛠️ Custom Architectures
*   **CUSTOM:** A blank slate for manual layer assembly, giving you full control over the topology.

---

## 💻 Quick Start (In-Model Pipeline Example)

This example shows how to use the **Surgical** approach to handle windowing and caching internally.

```cpp
#include "OpenWhiz/openwhiz.hpp"
#include <iostream>

int main() {
    ow::owNeuralNetwork nn;

    // 1. Setup Pure Data
    auto dataset = std::make_shared<ow::owDataset>();
    dataset->loadFromCSV("market_data.csv", true, true); // has_header, autoNormalize
    nn.setDataset(dataset);

    // 2. Build In-Model Pipeline
    nn.addLayer(std::make_shared<ow::owNormalizationLayer>(dataset->getInputVariableNum()));
    nn.addLayer(std::make_shared<ow::owSlidingWindowLayer>(10)); // 10-step windows
    nn.addLayer(std::make_shared<ow::owCacheLayer>(true));      // Enable cached shuffling

    // 3. Trainable Core
    nn.addLayer(std::make_shared<ow::owLinearLayer>(10, 16));
    nn.addLayer(std::make_shared<ow::owReLUActivation>());
    nn.addLayer(std::make_shared<ow::owLinearLayer>(16, 1));

    // 4. Train & Predict
    nn.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>());
    nn.train();

    auto prediction = nn.predict(); // Pipeline handles inverse normalization!
    std::cout << "Predicted Value: " << prediction(0, 0) << std::endl;

    return 0;
}
```

---

## 🛠️ Platform-Specific Benefits

*   **Financial Systems:** High-frequency model execution and multi-scale temporal analysis.
*   **Industrial/IoT:** Zero-dependency integration into real-time controllers with predictable performance.
*   **Desktop/HPC:** Massive throughput utilizing AVX-512/AVX2 for complex engineering simulations.
*   **Web/Mobile:** Lightweight binary size and efficient CPU usage for edge AI.

---

## 📜 License

This project is licensed under the **Apache License 2.0**.

---

## 🏛️ Developed By

**AITIAL Paris**
*Innovation in Artificial Intelligence and Industrial Automation.*
