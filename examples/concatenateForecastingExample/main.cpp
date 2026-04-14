#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

/**
 * @example concatenateForecastingExample
 * @brief Realistic USD/TRY Simulation using Multi-Scale View Architecture.
 */

int main() {
    std::cout << "=== OpenWhiz Multi-Scale USD/TRY Forecasting Simulation ===\n" << std::endl;

    const int masterWindow = 22;
    const int shortWindow = 5;

    auto dataset = std::make_shared<ow::owDataset>();
    ow::owNeuralNetwork nn;
    nn.setDataset(dataset);
    // CRITICAL: Ensure project type is set for automatic inverse normalization in predict()
    nn.setProjectType(ow::owProjectType::FORECASTING);

    // 1. Define Branches
    auto swShort = std::make_shared<ow::owSlidingWindowLayer>(shortWindow, 1, masterWindow, true);
    swShort->setNeuronNum(2); 
    
    auto swLong = std::make_shared<ow::owSlidingWindowLayer>(masterWindow, 1, masterWindow, false);
    swLong->setNeuronNum(2);

    // 2. Concatenate
    auto concat = std::make_shared<ow::owConcatenateLayer>(
        std::vector<std::shared_ptr<ow::owLayer>>{swShort, swLong},
        true 
    );
    nn.addLayer(concat);

    size_t totalFeatures = swShort->getOutputSize() + swLong->getOutputSize();
    nn.addLayer(std::make_shared<ow::owLinearLayer>(totalFeatures, 64));
    nn.addLayer(std::make_shared<ow::owLinearLayer>(64, 32));
    nn.addLayer(std::make_shared<ow::owLinearLayer>(32, 1)); 

    // 3. Generate Realistic USD/TRY Data
    std::cout << "Generating Realistic USD/TRY Data..." << std::endl;
    const std::string csvFile = "examples/concatenateForecastingExample/usd_try_sim.csv";
    {
        std::ofstream file(csvFile);
        file << "Price,Volume,Target\n";
        float price = 30.0f;
        srand(42); // Fixed seed for reproducibility
        for (int i = 0; i < 500; ++i) {
            float trend = 0.015f; 
            float noise = ((rand() % 100) / 2000.0f) - 0.025f; 
            float volume = 1000.0f + (rand() % 500);
            
            float nextPrice = price + trend + noise;
            file << price << "," << volume << "," << nextPrice << "\n";
            price = nextPrice;
        }
    }

    // 4. Load and Prepare
    if (!dataset->loadFromCSV(csvFile, true, true)) return -1;
    dataset->setTargetVariableNum(1);
    dataset->prepareForecastData(masterWindow);
    dataset->setRatios(0.8f, 0.1f, 0.1f, true);

    // 5. Train
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.001f));
    nn.setLoss(std::make_shared<ow::owHuberLoss>(1.0f));
    nn.setPrintEpochInterval(100);
    nn.setMaximumEpochNum(500); // Increased epochs for better fit
    
    std::cout << "Training Multi-Scale Model..." << std::endl;
    nn.train();

    // 6. Final Results
    std::cout << "\n--- Final USD/TRY Predictions (Inverse Normalized) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    auto testIn = dataset->getTestInput();
    auto testOut = dataset->getTestTarget();
    size_t displayCount = testIn.shape()[0] > 5 ? 5 : testIn.shape()[0];

    for (size_t i = 0; i < displayCount; ++i) {
        ow::owTensor<float, 2> sample(1, testIn.shape()[1]);
        for(size_t j=0; j<testIn.shape()[1]; ++j) sample(0, j) = testIn(i, j);

        // predict() should now return values in TL range (30+)
        auto pred = nn.predict(sample);
        float predictedPrice = pred(0, 0);
        
        ow::owTensor<float, 2> actualTensor(1, 1);
        actualTensor(0, 0) = testOut(i, 0);
        dataset->inverseNormalize(actualTensor);
        float actualPrice = actualTensor(0, 0);

        std::cout << "Test Sample " << i << ": Predicted: " << std::setw(8) << predictedPrice 
                  << " TL | Actual: " << std::setw(8) << actualPrice 
                  << " TL | Err: " << std::abs(predictedPrice - actualPrice) << " TL" << std::endl;
    }

    return 0;
}
