#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== OpenWhiz USD/TRY Simplified Expert Training (Arbiter Fixed) ===\n" << std::endl;

    const std::string csvFile = "C:/dev/OpenWhiz/examples/tryusdForecastExample/usd-try_3years.csv";
    const int masterWindow = 22; 
    const int shortWindow = 5;   
    const int expertEpochs = 2000; 
    const int arbiterEpochs = 1000; // Calibrating weights only
    const float rateScale = 100.0f; 

    auto dataset = std::make_shared<ow::owDataset>();
    if (!dataset->loadFromCSV(csvFile, true, true)) return -1;

    dataset->setColumnUsage("Date", ow::ColumnUsage::UNUSED);
    dataset->setTargetVariableNum(1);
    dataset->prepareForecastData(masterWindow);
    dataset->setRatios(0.98f, 0.02f, 0.0f, false); 

    auto allIn = dataset->getTrainInput();
    auto allTarget = dataset->getTrainTarget();
    
    size_t expertSamples = 500;
    size_t expStart = allIn.shape()[0] > expertSamples ? allIn.shape()[0] - expertSamples : 0;
    size_t actualSamples = allIn.shape()[0] - expStart;

    ow::owTensor<float, 2> expertIn(actualSamples, allIn.shape()[1]);
    ow::owTensor<float, 2> expertTarget(actualSamples, allTarget.shape()[1]);
    for(size_t i = 0; i < actualSamples; ++i) {
        for(size_t j = 0; j < allIn.shape()[1]; ++j) expertIn(i, j) = allIn(expStart + i, j);
        for(size_t j = 0; j < allTarget.shape()[1]; ++j) expertTarget(i, j) = allTarget(expStart + i, j);
    }

    ow::owNeuralNetwork nn;
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.001f));
    nn.setLoss(std::make_shared<ow::owHuberLoss>(1.0f));
    nn.setDataset(dataset);
    nn.setProjectType(ow::owProjectType::FORECASTING);

    int targetColIdx = dataset->getTargetColumnIndex(0);
    auto normP = dataset->getNormalizationParams(targetColIdx);
    float minV = normP.first;
    float maxV = normP.second;
    float rangeV = (maxV - minV == 0) ? 1.0f : (maxV - minV);

    // --- BRANCH 1: WEEKLY ---
    auto branch1 = std::make_shared<ow::owSequentialLayer>();
    auto swShort = std::make_shared<ow::owSlidingWindowLayer>(shortWindow, 1, masterWindow, false);
    swShort->setNeuronNum(1); 
    branch1->addLayer(swShort);
    branch1->addLayer(std::make_shared<ow::owChangeRateLayer>(shortWindow)); 
    auto affine1 = std::make_shared<ow::owAffineLayer>();
    affine1->fromXML("<A>" + std::to_string(rateScale) + "</A><B>0.0</B>");
    affine1->setFrozen(true); 
    branch1->addLayer(affine1);
    branch1->addLayer(std::make_shared<ow::owLinearLayer>(shortWindow - 1, 64));
    branch1->addLayer(std::make_shared<ow::owLinearLayer>(64, 32));
    branch1->addLayer(std::make_shared<ow::owLinearLayer>(32, 1)); 
    branch1->setIndependentExpertMode(true); 
    branch1->setLayerName("Weekly");

    // --- BRANCH 2: MONTHLY ---
    auto branch2 = std::make_shared<ow::owSequentialLayer>();
    auto swLong = std::make_shared<ow::owSlidingWindowLayer>(masterWindow, 1, masterWindow, false);
    swLong->setNeuronNum(1);
    branch2->addLayer(swLong);
    branch2->addLayer(std::make_shared<ow::owChangeRateLayer>(masterWindow));
    auto affine2 = std::make_shared<ow::owAffineLayer>();
    affine2->fromXML("<A>" + std::to_string(rateScale) + "</A><B>0.0</B>");
    affine2->setFrozen(true); 
    branch2->addLayer(affine2);
    branch2->addLayer(std::make_shared<ow::owLinearLayer>(masterWindow - 1, 64));
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
    nn.addLayer(mergeLayer);

    auto opt1 = new ow::owADAMOptimizer(0.001f);
    auto opt2 = new ow::owADAMOptimizer(0.001f);
    branch1->setOptimizer(opt1);
    branch2->setOptimizer(opt2);

    ow::owTensor<float, 2> expertTargetRateScaled(actualSamples, 1);
    for(size_t i = 0; i < actualSamples; ++i) {
        float p_prev_tl = expertIn(i, masterWindow - 1) * rangeV + minV;
        float p_curr_tl = expertTarget(i, 0) * rangeV + minV;
        expertTargetRateScaled(i, 0) = ((p_curr_tl - p_prev_tl) / p_prev_tl) * rateScale; 
    }

    std::cout << "Phase 1: Training Experts..." << std::endl;
    branch1->setTarget(&expertTargetRateScaled);
    branch2->setTarget(&expertTargetRateScaled);
    mergeLayer->setFrozen(true); 
    for(int epoch = 1; epoch <= expertEpochs; ++epoch) {
        branch1->reset(); branch2->reset();
        branch1->forward(expertIn); branch2->forward(expertIn);
        branch1->trainIndependentExpertOnly(); branch2->trainIndependentExpertOnly();
    }

    std::cout << "Phase 2: Calibrating Arbiter..." << std::endl;
    branch1->setFrozen(true); branch2->setFrozen(true);
    mergeLayer->setFrozen(false); 
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.001f)); 
    nn.partialFit(expertIn, expertTargetRateScaled, arbiterEpochs);

    // --- ARBITER FIX: Force Pure Weighted Ensemble ---
    float* arbParams = mergeLayer->getParamsPtr();
    float w1 = arbParams[0];
    float w2 = arbParams[1];
    float totalW = std::abs(w1) + std::abs(w2);
    if(totalW < 1e-5f) { w1 = 0.5f; w2 = 0.5f; totalW = 1.0f; }
    
    arbParams[0] = w1 / totalW; // Normalized weight 1
    arbParams[1] = w2 / totalW; // Normalized weight 2
    arbParams[2] = 0.0f;        // ZERO BIAS - Prevents "pulling down"

    std::cout << "Arbiter Weights Finalized: Weekly=" << (arbParams[0]*100.0f) 
              << "%, Monthly=" << (arbParams[1]*100.0f) << "%" << std::endl;

    auto fullData = dataset->getData();
    size_t totalRows = dataset->getSampleNum();
    size_t startIdx = totalRows - 11; 

    ow::owTensor<float, 2> currentInput(1, masterWindow);
    for (size_t j = 0; j < (size_t)masterWindow; ++j) currentInput(0, j) = fullData(startIdx, j);

    std::cout << "\n--- Historical Backtest (Arbiter Pull-Down Fixed) ---" << std::endl;
    std::cout << "Format: Day: [Final TL] | [Weekly TL] | [Monthly TL] | [ACTUAL TL]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 1; i <= 5; ++i) {
        float lastPriceTL = currentInput(0, masterWindow - 1) * rangeV + minV;

        auto predScaledRateTensor = nn.forward(currentInput);
        float predictedRate = predScaledRateTensor(0, 0) / rateScale; 
        
        auto b1ScaledRateT = branch1->forward(currentInput); 
        auto b2ScaledRateT = branch2->forward(currentInput);
        float b1Rate = b1ScaledRateT(0, 0) / rateScale; 
        float b2Rate = b2ScaledRateT(0, 0) / rateScale; 
        
        float finalPredTL = lastPriceTL * (1.0f + predictedRate);
        float b1TL = lastPriceTL * (1.0f + b1Rate);
        float b2TL = lastPriceTL * (1.0f + b2Rate);
        float actualTL = fullData(startIdx + i, targetColIdx) * rangeV + minV;

        std::cout << "Day " << i << ": " << finalPredTL << " | " << b1TL << " | " << b2TL << " | " << actualTL << " TL" << std::endl;

        float nextNorm = (finalPredTL - minV) / rangeV;
        for (size_t j = 0; j < (size_t)masterWindow - 1; ++j) currentInput(0, j) = currentInput(0, j + 1);
        currentInput(0, masterWindow - 1) = nextNorm;
    }

    delete opt1; delete opt2;
    return 0;
}
