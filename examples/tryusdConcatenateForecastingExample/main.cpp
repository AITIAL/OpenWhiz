#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== OpenWhiz USD/TRY Trend-Optimized Expert Training ===\n" << std::endl;

    const std::string csvFile = "C:/dev/OpenWhiz/examples/tryusdForecastExample/usd-try_3years.csv";
    const int masterWindow = 5;
    const int shortWindow = 5;
    const int expertEpochs = 2000; 
    const int arbiterEpochs = 1500; 

    auto dataset = std::make_shared<ow::owDataset>();
    if (!dataset->loadFromCSV(csvFile, true, true)) return -1;

    dataset->setColumnUsage("Date", ow::ColumnUsage::UNUSED);
    dataset->setTargetVariableNum(1);
    dataset->prepareForecastData(masterWindow);
    dataset->setRatios(0.98f, 0.02f, 0.0f, false); 

    auto allIn = dataset->getTrainInput();
    auto allTarget = dataset->getTrainTarget();
    
    // Focus on recent trend (Last 500 samples)
    size_t expertSamples = 500;
    size_t expStart = allIn.shape()[0] > expertSamples ? allIn.shape()[0] - expertSamples : 0;
    ow::owTensor<float, 2> expertIn(allIn.shape()[0] - expStart, allIn.shape()[1]);
    ow::owTensor<float, 2> expertTarget(allTarget.shape()[0] - expStart, allTarget.shape()[1]);
    for(size_t i = 0; i < expertIn.shape()[0]; ++i) {
        for(size_t j = 0; j < allIn.shape()[1]; ++j) expertIn(i, j) = allIn(expStart + i, j);
        for(size_t j = 0; j < allTarget.shape()[1]; ++j) expertTarget(i, j) = allTarget(expStart + i, j);
    }

    ow::owNeuralNetwork nn;
    // Start with Adam for overall stability
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.001f));
    nn.setLoss(std::make_shared<ow::owHuberLoss>(1.0f));
    nn.setDataset(dataset);
    nn.setProjectType(ow::owProjectType::FORECASTING);

    // --- BRANCH 1: WEEKLY ---
    auto branch1 = std::make_shared<ow::owSequentialLayer>();
    auto swShort = std::make_shared<ow::owSlidingWindowLayer>(shortWindow, 1, masterWindow, true);
    swShort->setNeuronNum(1); 
    branch1->addLayer(swShort);
    branch1->addLayer(std::make_shared<ow::owTrendLayer>(swShort->getOutputSize()));
    branch1->addLayer(std::make_shared<ow::owLinearLayer>(swShort->getOutputSize(), 64));
    branch1->addLayer(std::make_shared<ow::owLinearLayer>(64, 32));
    branch1->addLayer(std::make_shared<ow::owLinearLayer>(32, 1)); 
    branch1->setIndependentExpertMode(true); 
    branch1->setLayerName("Weekly");

    // --- BRANCH 2: MONTHLY ---
    auto branch2 = std::make_shared<ow::owSequentialLayer>();
    auto swLong = std::make_shared<ow::owSlidingWindowLayer>(masterWindow, 1, masterWindow, false);
    swLong->setNeuronNum(1);
    branch2->addLayer(swLong);
    branch2->addLayer(std::make_shared<ow::owTrendLayer>(swLong->getOutputSize()));
    branch2->addLayer(std::make_shared<ow::owLinearLayer>(swLong->getOutputSize(), 64));
    branch2->addLayer(std::make_shared<ow::owLinearLayer>(64, 32));
    branch2->addLayer(std::make_shared<ow::owLinearLayer>(32, 1)); 
    branch2->setIndependentExpertMode(true); 
    branch2->setLayerName("Monthly");

    auto concat = std::make_shared<ow::owConcatenateLayer>();
    concat->setUseSharedInput(true);
    concat->addBranch(branch1);
    concat->addBranch(branch2);
    nn.addLayer(concat);

    auto mergeLayer = std::make_shared<ow::owLinearLayer>(2, 1);
    mergeLayer->getParamsPtr()[0] = 0.5f; 
    mergeLayer->getParamsPtr()[1] = 0.5f;
    mergeLayer->getParamsPtr()[2] = 0.0f;
    nn.addLayer(mergeLayer);

    // --- PHASE 1 ---
    std::cout << "Phase 1: Training Trend-Augmented Experts (" << expertEpochs << " Epochs)..." << std::endl;
    branch1->setTarget(&expertTarget);
    branch2->setTarget(&expertTarget);
    mergeLayer->setFrozen(true); 
    for(int epoch = 1; epoch <= expertEpochs; ++epoch) {
        branch1->reset(); branch2->reset();
        branch1->forward(expertIn); branch2->forward(expertIn);
        branch1->trainIndependentExpertOnly(); branch2->trainIndependentExpertOnly();
    }

    // --- PHASE 2 ---
    std::cout << "Phase 2: Final Arbiter Calibration (ADAM) (" << arbiterEpochs << " Epochs)..." << std::endl;
    branch1->setFrozen(true); branch2->setFrozen(true);
    mergeLayer->setFrozen(false); 
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.001f)); 
    nn.setMaximumEpochNum(arbiterEpochs);
    nn.setPrintEpochInterval(500);
    nn.train();

    // Verification
    auto fullData = dataset->getData();
    size_t totalRows = dataset->getSampleNum();
    size_t startIdx = totalRows - 6; 
    size_t targetColIdx = fullData.shape()[1] - 1;

    size_t networkInputDim = nn.getLayers()[0]->getInputSize();
    ow::owTensor<float, 2> currentInput(1, networkInputDim);
    for (size_t j = 0; j < networkInputDim; ++j) currentInput(0, j) = fullData(startIdx, j);

    std::cout << "\n--- Trend-Augmented Forecast Comparison ---" << std::endl;
    std::cout << "Format: Day: [Final] | [Weekly] | [Monthly] | [ACTUAL]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 1; i <= 5; ++i) {
        auto predNormalized = nn.forward(currentInput);
        float nextPredRaw = predNormalized(0, 0);

        ow::owTensor<float, 2> pT(1, 1), b1T(1, 1), b2T(1, 1), aT(1, 1);
        pT(0, 0) = nextPredRaw; 
        b1T = branch1->forward(currentInput); 
        b2T = branch2->forward(currentInput);
        aT(0, 0) = fullData(startIdx + i, targetColIdx);

        dataset->inverseNormalize(pT); dataset->inverseNormalize(b1T); dataset->inverseNormalize(b2T); dataset->inverseNormalize(aT);
        std::cout << "Day " << i << ": " << pT(0, 0) << " | " << b1T(0, 0) << " | " << b2T(0, 0) << " | " << aT(0, 0) << " TL" << std::endl;

        for (size_t j = 0; j < (size_t)masterWindow - 1; ++j) currentInput(0, j) = currentInput(0, j + 1);
        currentInput(0, masterWindow - 1) = nextPredRaw;
    }

    return 0;
}
