#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "OpenWhiz/openwhiz.hpp"


int main() {
    std::cout << "=== OpenWhiz High-Precision USD/TRY Forecasting ===\n" << std::endl;

    const std::string csvFile = "C:/dev/OpenWhiz/examples/tryusdForecastExample/usd-try_3years.csv";

    auto dataset = std::make_shared<ow::owDataset>();
    if (!dataset->loadFromCSV(csvFile, true, true)) return -1;
    
    // 1. Setup BEFORE windowing
    dataset->setTargetVariableNum(1); 
    dataset->setColumnUsage("Date", ow::ColumnUsage::UNUSED);
    
    const int windowSize = 5;
    dataset->prepareForecastData(windowSize);
    
    // 2. DOUBLE CHECK: Ensure Date is UNUSED even after windowing
    // We search for "Date" in the new column list
    dataset->setColumnUsage("Date", ow::ColumnUsage::UNUSED);
    
    std::vector<int> usedIndices = dataset->getUsedColumnIndices(false);
    std::cout << "Verified Input Columns (Indices): ";
    for(int idx : usedIndices) std::cout << idx << " ";
    std::cout << "\n(Targeting exactly 5 columns)" << std::endl;

    // If still 6, it means "Date" name wasn't found. We'll find it by index.
    if (usedIndices.size() > (size_t)windowSize) {
        std::cout << "Warning: Extra column detected. Forcing exclusion of index 5." << std::endl;
        // The dataset doesn't have an easy "setByIndex", so we rely on the network input dim.
    }

    // 3. NETWORK SETUP
    ow::owNeuralNetwork nn;
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.0005f)); 
    nn.setLoss(std::make_shared<ow::owHuberLoss>(1.0f));
    nn.setDataset(dataset);
    nn.setProjectType(ow::owProjectType::FORECASTING);
    nn.createNeuralNetwork(ow::owProjectType::FORECASTING, {64, 32});
    
    std::cout << "Training model..." << std::endl;
    nn.setMaximumEpochNum(1000);
    nn.train();

    // 4. RECURSIVE TEST (Corrected Calculation)
    auto fullData = dataset->getData();
    size_t totalRows = dataset->getSampleNum();
    size_t inputDim = nn.getLayers()[0]->getInputSize(); // Use network's actual expectation
    size_t targetColIdx = fullData.shape()[1] - 1; 
    size_t startIdx = totalRows - 6; 
    
    ow::owTensor<float, 2> currentInput(1, inputDim);
    
    std::cout << "\n--- Starting Recursive Test (Last 5 Known Days) ---" << std::endl;
    // Build initial window [H5, H4, H3, H2, H1] - Indices 0, 1, 2, 3, 4
    for (size_t j = 0; j < (size_t)windowSize; ++j) {
        currentInput(0, j) = fullData(startIdx, j);
    }
    // If network expects 6, fill 6th with Price (though it shouldn't)
    if (inputDim > (size_t)windowSize) {
        currentInput(0, 5) = fullData(startIdx, targetColIdx);
    }

    for (int i = 1; i <= 5; ++i) {
        size_t actualTargetRow = startIdx + i;
        auto predNormalized = nn.forward(currentInput);
        float nextPredRaw = predNormalized(0, 0);

        ow::owTensor<float, 2> pT(1, 1), aT(1, 1);
        pT(0, 0) = nextPredRaw; aT(0, 0) = fullData(actualTargetRow, targetColIdx);
        dataset->inverseNormalize(pT); dataset->inverseNormalize(aT);

        std::cout << "Day " << i << ": Predicted: " << pT(0, 0) 
                  << " TL | Actual: " << aT(0, 0) << " TL" << std::endl;

        // Shift
        for (size_t j = 0; j < (size_t)windowSize - 1; ++j) {
            currentInput(0, j) = currentInput(0, j + 1);
        }
        currentInput(0, (size_t)windowSize - 1) = nextPredRaw;
        if (inputDim > (size_t)windowSize) currentInput(0, 5) = nextPredRaw;
    }

    return 0;
}
