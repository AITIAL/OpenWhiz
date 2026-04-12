/*
 * owNeuralNetwork.inl
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once

namespace ow {

inline owNeuralNetwork::owNeuralNetwork() {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    m_rng.seed(static_cast<unsigned int>(seed));
    m_dataset = std::make_shared<owDataset>();
    m_optimizer = std::make_shared<owLBFGSOptimizer>(); // Default optimizer
    m_loss = std::make_shared<owMeanSquaredErrorLoss>(); // Default loss
}

inline void owNeuralNetwork::setDataset(std::shared_ptr<owDataset> ds) { m_dataset = ds; }
inline owDataset* owNeuralNetwork::getDataset() { return m_dataset.get(); }
inline bool owNeuralNetwork::loadData(const std::string& filename) { return m_dataset->loadFromCSV(filename); }

inline void owNeuralNetwork::setRegularization(int type) { m_regType = type; for (auto& layer : m_layers) layer->setRegularization(type); }
inline void owNeuralNetwork::addLayer(std::shared_ptr<owLayer> layer) {
    if (layer) { 
        layer->setOptimizer(m_optimizer.get()); 
        layer->setRegularization(m_regType); 
        m_layers.push_back(layer); 
    }
}

inline std::shared_ptr<owOptimizer> owNeuralNetwork::getOptimizer() { 
    if (!m_optimizer) m_optimizer = std::make_shared<owLBFGSOptimizer>();
    return m_optimizer; 
}

inline void owNeuralNetwork::setOptimizer(std::shared_ptr<owOptimizer> opt) { 
    m_optimizer = opt; 
    for (auto& l : m_layers) l->setOptimizer(m_optimizer.get()); 
}

inline std::vector<std::shared_ptr<owLayer>> owNeuralNetwork::getLayers() { return m_layers; }
inline void owNeuralNetwork::setLoss(std::shared_ptr<owLoss> loss) { m_loss = loss; }

inline owTensor<float, 2> owNeuralNetwork::forward(const owTensor<float, 2>& input) {
    owTensor<float, 2> currentOutput = input;
    for (auto& layer : m_layers) currentOutput = layer->forward(currentOutput);
    return currentOutput;
}

inline void owNeuralNetwork::backward(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) {
    owTensor<float, 2> grad = m_loss->gradient(prediction, target);
    
    // Global Gradient Clipping: Scale down if the gradient norm is too high
    // This helps stability when targets are large (like in airfoil noise)
    float gradNormSq = 0;
    for (size_t i = 0; i < grad.size(); ++i) gradNormSq += grad.data()[i] * grad.data()[i];
    float gradNorm = std::sqrt(gradNormSq);
    if (gradNorm > 10.0f) {
        float scale = 10.0f / gradNorm;
        for (size_t i = 0; i < grad.size(); ++i) grad.data()[i] *= scale;
    }

    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) grad = (*it)->backward(grad);
}

inline void owNeuralNetwork::trainStep() { for (auto& layer : m_layers) layer->train(); }

inline void owNeuralNetwork::reset() { for (auto& layer : m_layers) layer->reset(); }

inline size_t owNeuralNetwork::getTotalParameterCount() const {
    size_t total = 0;
    for (const auto& layer : m_layers) total += layer->getParamsCount();
    return total;
}

inline void owNeuralNetwork::getGlobalParameters(owTensor<float, 1>& target) const {
    size_t offset = 0;
    for (const auto& layer : m_layers) {
        size_t count = layer->getParamsCount();
        if (count > 0) {
            std::copy(layer->getParamsPtr(), layer->getParamsPtr() + count, target.data() + offset);
            offset += count;
        }
    }
}

inline void owNeuralNetwork::setGlobalParameters(const owTensor<float, 1>& source) {
    size_t offset = 0;
    for (auto& layer : m_layers) {
        size_t count = layer->getParamsCount();
        if (count > 0) {
            std::copy(source.data() + offset, source.data() + offset + count, layer->getParamsPtr());
            offset += count;
            layer->synchronize();
        }
    }
}

inline void owNeuralNetwork::getGlobalGradients(owTensor<float, 1>& target) const {
    size_t offset = 0;
    for (const auto& layer : m_layers) {
        size_t count = layer->getParamsCount();
        if (count > 0) {
            std::copy(layer->getGradsPtr(), layer->getGradsPtr() + count, target.data() + offset);
            offset += count;
        }
    }
}

inline float owNeuralNetwork::calculateLoss(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) {
    return m_loss ? m_loss->compute(prediction, target) : 0.0f;
}

inline owTensor<std::string, 1> owNeuralNetwork::getLayerNames() const {
    owTensor<std::string, 1> res(m_layers.size());
    for (size_t i = 0; i < m_layers.size(); ++i) {
        res(i) = m_layers[i]->getLayerName();
    }
    return res;
}

inline owTensor<float, 1> owNeuralNetwork::getNeuronNums() const {
    owTensor<float, 1> res(m_layers.size());
    for (size_t i = 0; i < m_layers.size(); ++i) {
        res(i) = (float)m_layers[i]->getNeuronNum();
    }
    return res;
}

inline void owNeuralNetwork::partialFit(const owTensor<float, 2>& input, const owTensor<float, 2>& target, int steps) {
    for (int i = 0; i < steps; ++i) {
        auto pred = forward(input);
        backward(pred, target);
        trainStep();
    }
    m_isPartiallyFitted = true;
}

inline bool owNeuralNetwork::saveToXML(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << "<owNeuralNetwork>\n";
    file << "  <Optimizer type=\"" << (m_optimizer ? m_optimizer->getOptimizerName() : "ADAM") 
         << "\" LR=\"" << (m_optimizer ? m_optimizer->getLearningRate() : 0.01f) << "\" />\n";
    file << "  <Loss type=\"" << (m_loss ? m_loss->getLossName() : "Mean Squared Error Loss") << "\" />\n";
    
    file << "  <Hyperparameters>\n";
    file << "    <ProjectType>" << static_cast<int>(m_projectType) << "</ProjectType>\n";
    file << "    <MaxEpochs>" << m_maxEpochs << "</MaxEpochs>\n";
    file << "    <MaxTime>" << m_maxTime << "</MaxTime>\n";
    file << "    <MinError>" << m_minError << "</MinError>\n";
    file << "    <LossStagnationTolerance>" << m_lossStagnationTolerance << "</LossStagnationTolerance>\n";
    file << "    <LossStagnationPatience>" << m_lossStagnationPatience << "</LossStagnationPatience>\n";
    file << "    <RegType>" << m_regType << "</RegType>\n";
    file << "    <PrintInterval>" << m_printInterval << "</PrintInterval>\n";
    file << "  </Hyperparameters>\n";

    file << "  <Layers count=\"" << m_layers.size() << "\">\n";
    for (const auto& layer : m_layers) {
        file << "    <Layer type=\"" << layer->getLayerName() << "\">\n";
        file << layer->toXML();
        file << "    </Layer>\n";
    }
    file << "  </Layers>\n";
    file << "</owNeuralNetwork>\n";
    file.close();
    m_isPartiallyFitted = false;
    return true;
}

/**
 * @brief Helper factory to create optimizers by name.
 */
inline std::shared_ptr<owOptimizer> createOptimizerByName(const std::string& name) {
    if (name == "ADAM") return std::make_shared<owADAMOptimizer>();
    if (name == "SGD") return std::make_shared<owSGDOptimizer>();
    if (name == "RMSProp") return std::make_shared<owRMSPropOptimizer>();
    if (name == "Momentum") return std::make_shared<owMomentumOptimizer>();
    if (name == "L-BFGS") return std::make_shared<owLBFGSOptimizer>();
    if (name == "Conjugate Gradient") return std::make_shared<owConjugateGradientOptimizer>();
    return std::make_shared<owADAMOptimizer>();
}

/**
 * @brief Helper factory to create loss functions by name.
 */
inline std::shared_ptr<owLoss> createLossByName(const std::string& name) {
    if (name == "Mean Squared Error Loss") return std::make_shared<owMeanSquaredErrorLoss>();
    if (name == "Mean Absolute Error Loss") return std::make_shared<owMeanAbsoluteErrorLoss>();
    if (name == "Huber Loss") return std::make_shared<owHuberLoss>();
    if (name == "Binary Cross-Entropy Loss") return std::make_shared<owBinaryCrossEntropyLoss>();
    if (name == "Categorical Cross-Entropy Loss") return std::make_shared<owCategoricalCrossEntropyLoss>();
    if (name == "Pinball Loss") return std::make_shared<owPinballLoss>();
    if (name == "Weighted Mean Squared Error Loss") return std::make_shared<owWeightedMeanSquaredErrorLoss>();
    if (name == "Margin Ranking Loss") return std::make_shared<owMarginRankingLoss>();
    return std::make_shared<owMeanSquaredErrorLoss>();
}

/**
 * @brief Helper factory to create layers by type name.
 */
inline std::shared_ptr<owLayer> createLayerByName(const std::string& type, size_t inputSize = 0) {
    if (type == "Linear Layer") return std::make_shared<owLinearLayer>(inputSize, 1);
    if (type == "Normalization Layer") return std::make_shared<owNormalizationLayer>(inputSize);
    if (type == "Inverse Normalization Layer") return std::make_shared<owInverseNormalizationLayer>(inputSize);
    if (type == "Probability Layer") return std::make_shared<owProbabilityLayer>();
    if (type == "LSTM Layer") return std::make_shared<owLSTMLayer>(inputSize, 1);
    if (type == "Smoothing Layer") return std::make_shared<owSmoothingLayer>();
    if (type == "Rescaling Layer") return std::make_shared<owRescalingLayer>(1.0f, 0.0f);
    if (type == "Ranking Layer") return std::make_shared<owRankingLayer>(inputSize > 0 ? inputSize : 1);
    if (type == "Quantile Layer") return std::make_shared<owQuantileLayer>();
    if (type == "Principal Component Analysis Layer") return std::make_shared<owPrincipalComponentAnalysisLayer>(inputSize > 0 ? inputSize : 1, 1);
    if (type == "Projection Layer") return std::make_shared<owProjectionLayer>(inputSize > 0 ? inputSize : 1, 1);
    if (type == "Distance Layer") return std::make_shared<owDistanceLayer>(inputSize > 0 ? inputSize : 1, 1);
    if (type == "Position Encoding Layer") return std::make_shared<owPositionEncodingLayer>(1, 1);
    if (type == "Multi-Head Attention Layer") return std::make_shared<owMultiHeadAttentionLayer>(1, 1);
    if (type == "DateTime Encoding Layer") return std::make_shared<owDateTimeEncodingLayer>();
    if (type == "Cluster Layer") return std::make_shared<owClusterLayer>(inputSize > 0 ? inputSize : 1, 1);
    if (type == "Clipping Layer") return std::make_shared<owClippingLayer>(0.0f, 1.0f);
    if (type == "Bounding Layer") return std::make_shared<owBoundingLayer>(0.0f, 1.0f);
    if (type == "Attention Layer") return std::make_shared<owAttentionLayer>(inputSize > 0 ? inputSize : 1);
    if (type == "Sliding Window Layer") return std::make_shared<owSlidingWindowLayer>();
    if (type == "Anomaly Detection Layer") return std::make_shared<owAnomalyDetectionLayer>();
    if (type == "Affine Layer") return std::make_shared<owAffineLayer>();
    if (type == "Addition Layer") return std::make_shared<owAdditionLayer>(inputSize > 0 ? inputSize : 1);
    if (type == "Concatenate Layer") return std::make_shared<owConcatenateLayer>();
    
    return nullptr;
}

inline bool owNeuralNetwork::loadFromXML(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string xml = buffer.str();

    // Recursive layer parser
    std::function<std::shared_ptr<owLayer>(const std::string&, const std::string&)> parseLayer;
    parseLayer = [&](const std::string& layerTag, const std::string& layerContent) -> std::shared_ptr<owLayer> {
        std::string type = owLayer::getAttr(layerTag, "type");
        auto layer = createLayerByName(type, 1);
        if (!layer) return nullptr;

        if (type == "Concatenate Layer") {
            auto concat = std::static_pointer_cast<owConcatenateLayer>(layer);
            std::vector<std::shared_ptr<owLayer>> branches;
            
            std::string countStr = owLayer::getNestedTagContent(layerContent, "BranchCount");
            size_t branchCount = countStr.empty() ? 0 : std::stoul(countStr);

            for (size_t i = 0; i < branchCount; ++i) {
                std::string branchTag = "Branch_" + std::to_string(i);
                size_t bStart = layerContent.find("<" + branchTag);
                if (bStart != std::string::npos) {
                    size_t bEndTag = layerContent.find(">", bStart);
                    std::string subTag = layerContent.substr(bStart, bEndTag - bStart + 1);
                    std::string subContent = owLayer::getNestedTagContent(layerContent, branchTag);
                    branches.push_back(parseLayer(subTag, subContent));
                }
            }
            concat->setBranches(branches);
        }

        layer->fromXML(layerContent);
        return layer;
    };

    // Optimizer
    size_t optPos = xml.find("<Optimizer");
    if (optPos != std::string::npos) {
        std::string optLine = xml.substr(optPos, xml.find("/>", optPos) - optPos);
        std::string type = owLayer::getAttr(optLine, "type");
        float lr = std::stof(owLayer::getAttr(optLine, "LR"));
        if (!type.empty()) setOptimizer(createOptimizerByName(type));
        if (m_optimizer) m_optimizer->setLearningRate(lr);
    }

    // Loss
    size_t lossPos = xml.find("<Loss");
    if (lossPos != std::string::npos) {
        std::string lossLine = xml.substr(lossPos, xml.find("/>", lossPos) - lossPos);
        std::string type = owLayer::getAttr(lossLine, "type");
        if (!type.empty()) setLoss(createLossByName(type));
    }

    // Hyperparameters
    std::string hpContent = owLayer::getTagContent(xml, "Hyperparameters");
    if (!hpContent.empty()) {
        std::string val;
        val = owLayer::getTagContent(hpContent, "ProjectType"); if (!val.empty()) m_projectType = static_cast<owProjectType>(std::stoi(val));
        val = owLayer::getTagContent(hpContent, "MaxEpochs"); if (!val.empty()) m_maxEpochs = std::stoi(val);
        val = owLayer::getTagContent(hpContent, "MaxTime"); if (!val.empty()) m_maxTime = std::stod(val);
        val = owLayer::getTagContent(hpContent, "MinError"); if (!val.empty()) m_minError = std::stof(val);
        val = owLayer::getTagContent(hpContent, "LossStagnationTolerance"); if (!val.empty()) m_lossStagnationTolerance = std::stof(val);
        val = owLayer::getTagContent(hpContent, "LossStagnationPatience"); if (!val.empty()) m_lossStagnationPatience = std::stoi(val);
        val = owLayer::getTagContent(hpContent, "RegType"); if (!val.empty()) m_regType = std::stoi(val);
        val = owLayer::getTagContent(hpContent, "PrintInterval"); if (!val.empty()) m_printInterval = std::stoi(val);
    }

    m_layers.clear();
    std::string layersContent = owLayer::getTagContent(xml, "Layers");
    size_t pos = 0;
    while ((pos = layersContent.find("<Layer", pos)) != std::string::npos) {
        size_t lineEnd = layersContent.find(">", pos);
        std::string layerTag = layersContent.substr(pos, lineEnd - pos + 1);
        
        // Find matching </Layer> by counting nested <Layer tags
        size_t searchPos = lineEnd + 1;
        int depth = 1;
        size_t layerEnd = std::string::npos;
        while (depth > 0) {
            size_t nextOpen = layersContent.find("<Layer", searchPos);
            size_t nextClose = layersContent.find("</Layer>", searchPos);
            
            if (nextClose == std::string::npos) break; // Should not happen in valid XML
            
            if (nextOpen != std::string::npos && nextOpen < nextClose) {
                depth++;
                searchPos = nextOpen + 6;
            } else {
                depth--;
                if (depth == 0) layerEnd = nextClose;
                searchPos = nextClose + 8;
            }
        }

        if (layerEnd == std::string::npos) break;

        std::string layerContent = layersContent.substr(lineEnd + 1, layerEnd - (lineEnd + 1));
        
        auto layer = parseLayer(layerTag, layerContent);
        if (layer) addLayer(layer);
        
        pos = layerEnd + 8;
    }

    return true;
}

inline void owNeuralNetwork::createNeuralNetwork(const std::vector<int>& hiddenSizes, 
                                               const std::string& hiddenAct, 
                                               const std::string& outputAct,
                                               bool useNormalization) {
    m_projectType = owProjectType::CUSTOM;
    m_layers.clear();
    int inputSize = m_dataset->getInputVariableNum();
    int targetSize = m_dataset->getTargetVariableNum();

    if (useNormalization) {
        auto normLayer = std::make_shared<owNormalizationLayer>(inputSize);
        owTensor<float, 2> min(1, inputSize), max(1, inputSize);
        for (int i = 0; i < inputSize; ++i) {
            auto params = m_dataset->getNormalizationParams(i);
            min(0, i) = params.first;
            max(0, i) = params.second;
        }
        normLayer->setStatistics(min, max);
        addLayer(normLayer);
    }

    int currentIn = inputSize;
    for (int hSize : hiddenSizes) {
        auto layer = std::make_shared<owLinearLayer>(currentIn, hSize);
        layer->initializeWeightsWithRNG(m_rng); // Pass current RNG
        layer->setActivationByName(hiddenAct);
        addLayer(layer);
        currentIn = hSize;
    }

    auto outLayer = std::make_shared<owLinearLayer>(currentIn, targetSize);
    outLayer->initializeWeightsWithRNG(m_rng); // Pass current RNG
    outLayer->setActivationByName(outputAct);
    addLayer(outLayer);

    // 4. Trailing Normalization (if requested)
    if (useNormalization) {
        auto invNormLayer = std::make_shared<owInverseNormalizationLayer>(targetSize);
        owTensor<float, 2> min(1, targetSize), max(1, targetSize);
        for (int i = 0; i < targetSize; ++i) {
            // Target variables start after input variables
            auto params = m_dataset->getNormalizationParams(inputSize + i);
            min(0, i) = params.first;
            max(0, i) = params.second;
        }
        invNormLayer->setStatistics(min, max);
        addLayer(invNormLayer);
    }
}

inline void owNeuralNetwork::createNeuralNetwork(owProjectType type, const std::vector<int>& hiddenSizes, int windowSize) {
    m_projectType = type;
    m_layers.clear();
    int inputSize = m_dataset->getInputVariableNum();
    int targetSize = m_dataset->getTargetVariableNum();

    // 1. Optional Normalization Layer
    if (type == owProjectType::APPROXIMATION || type == owProjectType::FORECASTING || type == owProjectType::CLASSIFICATION || type == owProjectType::CLUSTERING || type == owProjectType::ANOMALY_DETECTION) {
        auto normLayer = std::make_shared<owNormalizationLayer>(inputSize);
        owTensor<float, 2> min(1, inputSize), max(1, inputSize);
        for (int i = 0; i < inputSize; ++i) {
            auto params = m_dataset->getNormalizationParams(i);
            min(0, i) = params.first;
            max(0, i) = params.second;
        }
        normLayer->setStatistics(min, max);
        addLayer(normLayer);
    }

    if (type == owProjectType::CLUSTERING) {
        int latentDim = hiddenSizes.empty() ? inputSize : hiddenSizes[0];
        int numClusters = targetSize; // Assume target variables match number of clusters for distance output

        // Projection
        addLayer(std::make_shared<owProjectionLayer>(inputSize, latentDim));
        
        // Clustering
        addLayer(std::make_shared<owClusterLayer>(latentDim, numClusters));
        
        // Distance calculation
        addLayer(std::make_shared<owDistanceLayer>(numClusters, numClusters));
        
        return;
    }

    if (type == owProjectType::ANOMALY_DETECTION) {
        int latentDim = hiddenSizes.empty() ? inputSize : hiddenSizes[0];
        
        // Projection
        addLayer(std::make_shared<owProjectionLayer>(inputSize, latentDim));
        
        // Anomaly Detection
        addLayer(std::make_shared<owAnomalyDetectionLayer>());
        
        return;
    }

    // 2. Windowing / Hidden Layers
    int currentIn = inputSize;

    if (type == owProjectType::FORECASTING) {
        auto swLayer = std::make_shared<owSlidingWindowLayer>(windowSize, 1, -1, true);
        addLayer(swLayer);
        currentIn = (int)swLayer->getOutputSize();
    }

    for (int hSize : hiddenSizes) {
        auto layer = std::make_shared<owLinearLayer>(currentIn, hSize);
        layer->initializeWeightsWithRNG(m_rng);
        layer->setActivationByName("ReLU");
        addLayer(layer);
        currentIn = hSize;
    }

    // 3. Output Layer
    auto outLayer = std::make_shared<owLinearLayer>(currentIn, targetSize);
    outLayer->initializeWeightsWithRNG(m_rng);
    
    if (type == owProjectType::CLASSIFICATION) {
        outLayer->setActivationByName("Sigmoid");
    } else {
        outLayer->setActivationByName("Identity");
    }
    addLayer(outLayer);

    // 4. Trailing Layer
    if (type == owProjectType::APPROXIMATION || type == owProjectType::FORECASTING) {
        auto invNormLayer = std::make_shared<owInverseNormalizationLayer>(targetSize);
        owTensor<float, 2> min(1, targetSize), max(1, targetSize);
        for (int i = 0; i < targetSize; ++i) {
            auto params = m_dataset->getNormalizationParams(inputSize + i);
            min(0, i) = params.first;
            max(0, i) = params.second;
        }
        invNormLayer->setStatistics(min, max);
        addLayer(invNormLayer);
    } else if (type == owProjectType::CLASSIFICATION) {
        if (targetSize > 1) {
            addLayer(std::make_shared<owProbabilityLayer>());
        }
    }
}

inline void owNeuralNetwork::train() {
    if (!m_dataset || !m_optimizer || !m_loss) return;
    if (m_optimizer->supportsGlobalOptimization()) m_optimizer->optimizeGlobal(this, m_dataset.get());
    else runStandardTrainingLoop();
}

inline void owNeuralNetwork::runStandardTrainingLoop() {
    auto trainIn = m_dataset->getTrainInput();
    auto trainTarget = m_dataset->getTrainTarget();
    auto startTime = std::chrono::high_resolution_clock::now();
    
    float bestLoss = std::numeric_limits<float>::max();
    int patienceCounter = 0;

    for (int epoch = 1; epoch <= m_maxEpochs; ++epoch) {
        reset(); 
        auto pred = forward(trainIn);
        float loss = calculateLoss(pred, trainTarget);
        backward(pred, trainTarget);
        trainStep();
        m_lastTrainLoss = loss;

        if (m_enablePrinting && epoch % m_printInterval == 0) {
            std::cout << "Epoch " << epoch << "/" << m_maxEpochs << " - Loss: " << loss << std::endl;
        }

        // Check for stagnation
        if (m_lossStagnationEnabled) {
            if (loss < bestLoss - m_lossStagnationTolerance) {
                bestLoss = loss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
            }

            if (patienceCounter >= m_lossStagnationPatience) {
                m_finishReason = "Loss Stagnation";
                m_actualEpochs = epoch;
                break;
            }
        }

        if (m_minError > 0 && m_lastTrainLoss <= m_minError) {
            m_finishReason = "Min Error";
            m_actualEpochs = epoch;
            break;
        }
        
        m_actualEpochs = epoch;
    }
}

inline EvaluationReport owNeuralNetwork::evaluatePerformance(const owTensor<float, 2>& input, const owTensor<float, 2>& target, float tolerance) {
    EvaluationReport report;
    reset(); // Clear state for evaluation
    auto pred = forward(input);
    size_t n = input.shape()[0];
    size_t outDim = target.shape()[1];
    float mse = 0, mape = 0;
    int correct = 0;
    for (size_t i = 0; i < n; ++i) {
        bool rowCorrect = true;
        for (size_t j = 0; j < outDim; ++j) {
            float p = pred(i, j), t = target(i, j), diff = p - t;
            mse += diff * diff;
            if (std::abs(t) > 1e-7f) mape += std::abs(diff / t);
            
            // Fix: If target is 0, use tolerance as an absolute threshold.
            // Otherwise, use it as a percentage of the target.
            float threshold = (std::abs(t) < 1e-7f) ? tolerance : std::abs(t * tolerance);
            if (std::abs(diff) > threshold) rowCorrect = false;
        }
        if (rowCorrect) correct++;
    }
    report.rmse = std::sqrt(mse / (n * outDim));
    report.mape = (mape / (n * outDim)) * 100.0f;
    report.accuracy = (float)correct / n;
    return report;
}

inline EvaluationReport owNeuralNetwork::evaluatePerformance(float tolerance) {
    if (!m_dataset) return EvaluationReport();
    auto testIn = m_dataset->getTestInput();
    auto testOut = m_dataset->getTestTarget();
    return evaluatePerformance(testIn, testOut, tolerance);
}

inline std::string owNeuralNetwork::predictLabel(const owTensor<float, 2>& input, int targetVarIdx) {
    auto pred = forward(input);
    if (!m_dataset) return "";
    int actualColIdx = m_dataset->getTargetColumnIndex(targetVarIdx);
    return m_dataset->getLabelName(actualColIdx, pred(0, targetVarIdx));
}

inline owTensor<float, 2> owNeuralNetwork::forecast(int steps) {
    if (!m_dataset) return owTensor<float, 2>(0, 0);
    return forecast(m_dataset->getLastSample(), steps);
}

inline owTensor<float, 2> owNeuralNetwork::forecast(const owTensor<float, 2>& initialSample, int steps) {
    if (steps <= 0) return owTensor<float, 2>(0, 0);
    
    // 1. Initial State
    reset(); 
    owTensor<float, 2> currentInput = initialSample;
    size_t inputFeatures = currentInput.shape()[1];
    
    // We run one forward pass to find out the target size
    auto firstPred = forward(currentInput);
    size_t targetSize = firstPred.shape()[1];
    
    owTensor<float, 2> results(steps, targetSize);
    
    // 2. Recursive Loop
    for (int i = 0; i < steps; ++i) {
        // Forward uses existing history in SlidingWindowLayer (since we didn't call reset inside loop)
        auto pred = forward(currentInput);
        
        // Store prediction
        for (size_t j = 0; j < targetSize; ++j) {
            results(i, j) = pred(0, j);
        }
        
        // Update input for next step (t+1)
        // If target size matches input size, replace entire input.
        // Otherwise, replace the LAST 'targetSize' columns (standard assumption for forecasting targets).
        if (targetSize == inputFeatures) {
            currentInput = pred;
        } else if (targetSize < inputFeatures) {
            size_t offset = inputFeatures - targetSize;
            for (size_t j = 0; j < targetSize; ++j) {
                currentInput(0, offset + j) = pred(0, j);
            }
        } else {
            // Target size > Input features: Just take what we can fit
            for (size_t j = 0; j < inputFeatures; ++j) {
                currentInput(0, j) = pred(0, j);
            }
        }
    }
    
    return results;
}

inline void owNeuralNetwork::printEvaluationReport(const EvaluationReport& report) const {
    std::cout << "--- Evaluation Report ---" << std::endl;
    std::cout << "RMSE: " << report.rmse << std::endl;
    std::cout << "MAPE: " << report.mape << "%" << std::endl;
    std::cout << "Accuracy (within tolerance): " << report.accuracy * 100.0f << "%" << std::endl;
}

inline std::shared_ptr<owActivation> owNeuralNetwork::createActivationByName(const std::string& name) {
    if (name == "ReLU") return std::make_shared<owReLUActivation>();
    if (name == "Sigmoid") return std::make_shared<owSigmoidActivation>();
    if (name == "Tanh") return std::make_shared<owTanhActivation>();
    if (name == "LeakyReLU") return std::make_shared<owLeakyReLUActivation>();
    return std::make_shared<owIdentityActivation>();
}

}
