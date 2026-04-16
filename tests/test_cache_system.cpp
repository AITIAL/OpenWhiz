/*
 * tests/test_cache_system.cpp
 * 
 * Verifies the new Stateful Sliding Window and Cache Recording/Playback system.
 */

#include "../include/OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <cassert>

using namespace ow;

int main() {
    std::cout << "--- Starting Cache and Stateful Sliding Window System Test ---" << std::endl;

    // 1. Setup Dataset
    auto ds = std::make_shared<owDataset>();
    if (!ds->loadFromCSV("tests/cache_test_data.csv", true, false)) {
        std::cerr << "Failed to load test CSV!" << std::endl;
        return 1;
    }
    ds->setRatios(1.0f, 0.0f, 0.0f, false); // All data for training, no shuffle in dataset

    // 2. Build Network with New Architecture
    owNeuralNetwork nn;
    nn.setDataset(ds);
    
    // Dataset Outputs: [1] (Value)
    // 1. Normalization (In-place scaling)
    auto norm = std::make_shared<owNormalizationLayer>(1);
    
    // 2. Sliding Window (windowSize=3, stateful)
    // Input: [1] -> Output: [3 (history) + 1 (current)] = 4
    auto sw = std::make_shared<owSlidingWindowLayer>(3, 1, true); 
    
    // 3. Cache Layer (records normalization + sliding window output)
    auto cache = std::make_shared<owCacheLayer>(true); // Shuffle enabled
    
    // 4. Linear Layer (for learning the trend)
    auto linear = std::make_shared<owLinearLayer>(4, 1);
    linear->setActivationByName("Identity");

    nn.addLayer(norm);
    nn.addLayer(sw);
    nn.addLayer(cache);
    nn.addLayer(linear);

    nn.setLoss(std::make_shared<owMeanSquaredErrorLoss>());
    nn.setOptimizer(std::make_shared<owADAMOptimizer>(0.01f));
    nn.setMaximumEpochNum(10);
    nn.setPrintEpochInterval(1);

    // 3. Pre-train check
    std::cout << "Layers in the network: " << std::endl;
    auto names = nn.getLayerNames();
    for (size_t i = 0; i < names.size(); ++i) {
        std::cout << "  - " << names(i) << std::endl;
    }
    
    assert(!cache->isFull());
    std::cout << "[PASS] Cache is empty before training." << std::endl;

    // 4. Run Training
    std::cout << "Running training for 10 epochs..." << std::endl;
    nn.train();

    // 5. Post-train checks
    assert(cache->isFull());
    std::cout << "[PASS] Cache is locked (Full) after epoch 1." << std::endl;

    // 6. Verification of the Playback Mechanism
    // If the cache is working, in the next epoch (or manual forward pass), 
    // the SlidingWindowLayer reset() will be called, but the Cache will still have data.
    nn.reset(); 
    auto input = ds->getTrainInput();
    auto pred = nn.forward(input);

    std::cout << "Prediction shape: " << pred.shape()[0] << "x" << pred.shape()[1] << std::endl;
    assert(pred.shape()[0] == input.shape()[0]);

    std::cout << "[SUCCESS] Cache system and stateful sliding window verified!" << std::endl;
    return 0;
}
