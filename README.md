# OpenWhiz ⚡

**OpenWhiz** is a high-performance, zero-dependency and header-only C++ deep learning AI library designed for seamless integration into Desktop, Mobile, Web (via WASM), and Industrial applications. Developed by **AITIAL Paris**, it focuses on speed and mathematical accuracy deployment.

---

## 🚀 Key Features

*   **Zero-Dependency:** No external libraries required (no BLAS, no Protobuf, no Eigen). Just include and compile.
*   **Header-Only:** Easy to integrate into any build system (CMake, Make, MSVC).
*   **CPU Optimized:** Leveraging SIMD instructions (AVX-512, AVX2, SSE, ARM NEON) and multi-core processing via OpenMP.
*   **C++14 Compliant:** Modern, safe, and compatible with established embedded and industrial toolchains.
*   **Lightweight & Fast:** Minimal memory footprint, making it ideal for mobile devices and real-time industrial controllers.
*   **Comprehensive Toolset:** Includes advanced layers (LSTM, Attention, PCA), various optimizers (Adam, L-BFGS), and statistical analysis tools.

---

## 🏗️ Architecture Overview

OpenWhiz is structured into several modular components:

*   **Core:** High-performance `owTensor` engine and `owNeuralNetwork` manager.
*   **Layers:** From standard `Linear` and `LSTMLayer` to specialized `AnomalyDetection` and `PrincipalComponentAnalysisLayer`.
*   **Optimizers:** First-order (SGD, Adam, RMSProp) and second-order (L-BFGS, Conjugate Gradient) methods.
*   **Data:** `owDataset` for CSV handling and `owStatistics` for dataset profiling.
*   **Activations & Losses:** A wide range of non-linearities (ReLU, Tanh, Sigmoid) and loss functions (MSE, Huber, Cross-Entropy).

---

## 🧩 Project Types & Deep Learning Paradigms

OpenWhiz simplifies network construction through high-level project types, categorized by their learning paradigm:

### 🎯 Supervised Learning
Requires target labels/values for training. The network learns a mapping from inputs to outputs.
*   **APPROXIMATION:** Continuous function fitting (Regression) for industrial modeling and physical simulations.
*   **FORECASTING:** Time-series prediction for financial markets, macroeconomic indicators, demand planning, and predictive maintenance of machinery to anticipate future states or failures.
*   **CLASSIFICATION:** Categorical prediction (Multi-class/Binary) for decision-making and pattern recognition.

### 🔍 Unsupervised Learning
Learns patterns, structures, or anomalies directly from input data without explicit target labels.
*   **CLUSTERING:** Grouping similar data points using projection and distance metrics (Latent Space Analysis).
*   **ANOMALY_DETECTION:** Identifying and suppressing outliers or suspicious patterns in data streams (Statistical Z-Score & Projection).

### 🛠️ Custom Architectures
For advanced users who require full control over the network topology.
*   **CUSTOM:** A blank slate that does not apply any automatic wrapping or layers. You can manually add any combination of layers, activations, and specialized components to build unique architectures from scratch.

---

## 💻 Quick Start (Forecasting Example)

Since OpenWhiz is header-only, simply add the `include` folder to your project path. This example shows how to predict the next value in a time-series using the automated **FORECASTING** project type and the new Data-Centric architecture.

```cpp
#include "OpenWhiz/openwhiz.hpp"
#include <iostream>

int main() {
    ow::owNeuralNetwork nn;

    // 1. Setup Data with In-Place Normalization
    auto dataset = std::make_shared<ow::owDataset>();
    dataset->loadFromCSV("dataset.csv", true, true); // true, true = has_header, autoNormalize
    nn.setDataset(dataset);

    // 2. Prepare Windowed Data at Dataset Level
    // Each row now contains history [t-5...t-1]
    dataset->prepareForecastData(5); 

    // 3. Build Architecture
    // createNeuralNetwork now creates a clean Linear MLP fitting the windowed input
    nn.createNeuralNetwork(ow::owProjectType::FORECASTING, {16, 16}); 
    
    // 4. Configure Training
    nn.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>());

    // 5. Train (Shuffle is now safe as windows are packaged within each sample)
    nn.train();

    // 6. Predict Actual Values
    // nn.predict() automatically performs inverse normalization for you!
    auto prediction = nn.predict(); 
    std::cout << "Predicted Next Value (Actual Scale): " << prediction(0, 0) << std::endl;

    return 0;
}
```

---

## 🛠️ Platform-Specific Benefits

*   **Desktop/HPC:** Utilizes AVX-512/AVX2 for massive throughput in complex engineering simulations.
*   **Mobile:** Small binary size and efficient CPU usage ensure minimal impact on battery life.
*   **Web:** Compiles perfectly with Emscripten for high-performance browser-based AI.
*   **Industrial/IoT:** Predictable memory usage and smooth activation functions (like Tanh) are ideal for real-time control systems and signal processing.

---

## 📜 License

This project is licensed under the **Apache License 2.0**.

---

## 🏛️ Developed By

**AITIAL Paris**
*Innovation in Artificial Intelligence and Industrial Automation.*
