#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generates a sine wave CSV for forecasting (2 columns for explicit input/target)
void generateSineWaveCSV(const std::string& filename, int points) {
    std::ofstream file(filename);
    file << "Input,Target\n";
    for (int i = 0; i < points; ++i) {
        float val = std::sin(2.0f * M_PI * i / 50.0f); // Period of 50 points
        float nextVal = std::sin(2.0f * M_PI * (i + 1) / 50.0f);
        file << val << "," << nextVal << "\n";
    }
    file.close();
}

int main() {
    std::cout << "=== OpenWhiz Sine Wave Forecasting Example ===\n" << std::endl;

    // 1. Generate sine wave data
    const std::string csvFile = "examples/forecastExample/sine_wave.csv";
    generateSineWaveCSV(csvFile, 300);
    std::cout << "Generated 300 points of sine wave data." << std::endl;

    // 2. Setup Dataset with In-Place Normalization
    auto dataset = std::make_shared<ow::owDataset>();
    dataset->setDelimiter(',');
    // autoNormalize = true
    if (!dataset->loadFromCSV(csvFile, true, true)) {
        std::cerr << "Failed to load CSV data." << std::endl;
        return -1;
    }
    
    dataset->setTargetVariableNum(1);
    
    // Prepare Windowed Data at Dataset Level
    const int windowSize = 10;
    dataset->prepareForecastData(windowSize);
    
    dataset->setRatios(0.8f, 0.1f, 0.1f, true);

    std::cout << "Input variables (Windowed): " << dataset->getInputVariableNum() << std::endl;
    std::cout << "Target variables: " << dataset->getTargetVariableNum() << std::endl;

    // 3. Initialize Neural Network
    ow::owNeuralNetwork nn;
    nn.setDataset(dataset);
    
    // Simple MLP with 16 hidden neurons
    nn.createNeuralNetwork(ow::owProjectType::FORECASTING, {16});
    
    // Using L-BFGS for high precision on sine wave
    nn.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>(1.0f));
    nn.setMaximumEpochNum(200);
    nn.setLossStagnationTolerance(1e-6f);
    nn.setPrintEpochInterval(50);

    // 4. Train
    std::cout << "Training model to predict sine wave..." << std::endl;
    nn.train();

    // 5. Evaluation
    auto eval = nn.evaluatePerformance(0.05f);
    
    std::cout << "\nFinal Performance on Test Set:" << std::endl;
    nn.printEvaluationReport(eval);

    // 6. Predict Test (Using the new predict() API)
    std::cout << "\n--- Rolling Prediction Test (Last 5 points) ---" << std::endl;
    
    auto testIn = dataset->getTestInput();
    auto testOut = dataset->getTestTarget();
    size_t testSamples = testIn.shape()[0];
    
    for (size_t i = testSamples > 5 ? testSamples - 5 : 0; i < testSamples; ++i) {
        // Extract sample
        ow::owTensor<float, 2> sample(1, testIn.shape()[1]);
        for(size_t j=0; j<testIn.shape()[1]; ++j) sample(0, j) = testIn(i, j);

        // predict() handles forward + inverseNormalize
        auto pred = nn.predict(sample);
        
        // Recover actual value for comparison
        ow::owTensor<float, 2> actualTensor(1, 1);
        actualTensor(0, 0) = testOut(i, 0);
        dataset->inverseNormalize(actualTensor);
        float actualVal = actualTensor(0, 0);

        std::cout << "Step " << i << ": Predicted: " << pred(0, 0) 
                  << " | Actual: " << actualVal 
                  << " | Err: " << std::abs(pred(0, 0) - actualVal) << std::endl;
    }

    return 0;
}
