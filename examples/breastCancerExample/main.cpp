#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"


int main() {
    std::cout << "=== OpenWhiz Breast Cancer Classification Example ===\n" << std::endl;

    const std::string csvFile = "C:/dev/OpenWhiz/examples/breastCancerExample/breast_cancer.csv";

    ow::owNeuralNetwork nn;
    
    // 1. Setup Data
    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }

    // 2. Build Network Architecture
    nn.createNeuralNetwork(ow::owProjectType::CLASSIFICATION, {36, 36});
    
    // 3. Configure Training
//    nn.setLoss(std::make_shared<ow::owCategoricalCrossEntropyLoss>());
//    nn.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>());

    // 4. Train
    nn.train();

    // 5. Custom Classification Evaluation (Argmax Accuracy)
    auto testIn = nn.getDataset()->getTestInput();
    auto testOut = nn.getDataset()->getTestTarget();
    auto eval = nn.evaluatePerformance(testIn, testOut, 0.05f); // 5% tolerance
    
    std::cout << "\nFinal Performance on Test Set:" << std::endl;
    nn.printEvaluationReport(eval);

    // Logging extra stats before manual prediction
    std::cout << "Total Epochs: " << nn.getTrainingEpochNum() << std::endl;
    std::cout << "Total Time: " << nn.getTrainingTime() << "s" << std::endl;
    std::cout << "Final Train Error: " << nn.getLastTrainError() << std::endl;
    std::cout << "Final Val Error: " << nn.getLastValError() << std::endl;
    std::cout << "First Sample Type: " << nn.getDataset()->getSampleTypeString(0) << std::endl;

    // 6. Manual Prediction Check (Using RAW values directly!)
    // Input features: frequency;angle_of_attack;chord_lenght;velocity;thickness
    ow::owTensor<float, 2> input(2, 9);
    input.setValues({{5, 1, 1, 1, 2, 1, 3, 1, 1}, {8, 10, 10, 8, 7, 10, 9, 7, 1}});

    auto pred = nn.forward(input);

    std::cout << "Prediction for [5, 1, 1, 1, 2, 1, 3, 1, 1] = " << pred(0, 0) << " (Actual: 0)" << std::endl;
    std::cout << "Prediction for [8, 10, 10, 8, 7, 10, 9, 7, 1] = " << pred(1, 0) << " (Actual: 1)" << std::endl;

    return 0;
}
