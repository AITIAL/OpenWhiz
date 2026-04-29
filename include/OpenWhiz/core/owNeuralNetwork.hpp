/*
 * owNeuralNetwork.hpp
 *
 *  Created on: Nov 24, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <functional>
#include "owTensor.hpp"

/**
 * @file owNeuralNetwork.hpp
 * @brief Core neural network orchestration for the OpenWhiz AI library.
 */

namespace ow {

class owLayer;
class owOptimizer;
class owLoss;
class owDataset;
class owActivation;

/**
 * @enum owProjectType
 * @brief Categorizes the neural network's task to automate architecture setup.
 */
enum class owProjectType {
    CUSTOM,         ///< Standard network without automatic wrapping.
    APPROXIMATION,  ///< Continuous function fitting with automatic scaling.
    FORECASTING,    ///< Time-series prediction (architecturally similar to approximation).
    CLASSIFICATION, ///< Categorical prediction with probability output.
    CLUSTERING,     ///< Grouping similar data points using projection and distance metrics.
    ANOMALY_DETECTION ///< Detects and suppresses anomalies in the input stream.
};

/**
 * @struct EvaluationReport
 * @brief Holds comprehensive performance metrics for model evaluation.
 * 
 * This structure captures both classification (Accuracy, Precision, Recall, F1) 
 * and regression (R-Squared, RMSE, MAPE) metrics. It is designed to be 
 * platform-independent, providing high-level insights into model quality.
 */
struct EvaluationReport {
    float accuracy = 0.0f;     ///< Ratio of correctly predicted observations.
    float precision = 0.0f;    ///< Precision for classification tasks.
    float recall = 0.0f;       ///< Recall/Sensitivity for classification tasks.
    float f1Score = 0.0f;      ///< Harmonic mean of precision and recall.
    float rSquared = 0.0f;     ///< Proportion of variance explained (Regression).
    float rmse = 0.0f;         ///< Root Mean Square Error (Regression).
    float mape = 0.0f;         ///< Mean Absolute Percentage Error (Regression).
    float realScaleMape = 0.0f; ///< MAPE calculated on de-normalized values.
    bool isClassification = false; ///< Flag indicating the nature of the task.
};

/**
 * @class owNeuralNetwork
 * @brief High-level API for building, training, and evaluating neural networks.
 * 
 * The owNeuralNetwork class serves as the central orchestrator in OpenWhiz. It manages 
 * layers, optimizers, and loss functions while providing a simplified interface for 
 * training loops. Unlike lower-level tensor operations, this class focuses on 
 * architectural management and high-level training workflows.
 * 
 * Unique Features:
 * - Hybrid Training: Supports both standard backpropagation and global optimization (e.g., L-BFGS).
 * - Automatic Normalization: Can inject normalization layers based on dataset statistics.
 * - Stagnation Detection: Early stopping based on loss tolerance and patience.
 * 
 * Platform Notes:
 * - Computer/Industrial: Leverages multi-threading via optimizers and layers.
 * - Mobile/Web: Optimized for low-memory footprint when using standard training loops.
 */
class owNeuralNetwork {
public:
    /**
     * @brief Constructs an empty neural network with a time-based seed.
     */
    owNeuralNetwork();
    ~owNeuralNetwork() = default;

    /**
     * @brief Sets the seed for the internal random number generator.
     * @param seed The unsigned integer seed.
     */
    void setSeed(unsigned int seed) { m_rng.seed(seed); }

    /**
     * @brief Provides access to the internal Mersenne Twister RNG.
     * @return Reference to the std::mt19937 engine.
     */
    std::mt19937& getRNG() { return m_rng; }

    /**
     * @brief Sets the dataset to be used for training and evaluation.
     * @param ds Shared pointer to an owDataset instance.
     */
    void setDataset(std::shared_ptr<owDataset> ds);

    /**
     * @brief Gets a raw pointer to the current dataset.
     * @return Pointer to owDataset.
     */
    owDataset* getDataset();

    /**
     * @brief Convenience method to load data from a CSV file directly into the internal dataset.
     * @param filename Path to the CSV file.
     * @return True if loading was successful.
     */
    bool loadData(const std::string& filename, bool hasHeader = true, bool autoNormalize = false);

    /**
     * @brief Sets the maximum number of epochs for the training process.
     * @param num Maximum epoch count.
     */
    void setMaximumEpochNum(int num) { m_maxEpochs = num; }

    /**
     * @brief Returns the maximum configured epoch number.
     * @return Integer epoch count.
     */
    int getMaximumEpochNum() const { return m_maxEpochs; }

    /**
     * @brief Automatically constructs a Multi-Layer Perceptron (MLP) architecture.
     * @param hiddenSizes Vector containing the number of neurons for each hidden layer.
     * @param hiddenAct Name of the activation function for hidden layers (e.g., "ReLU").
     * @param outputAct Name of the activation function for the output layer (e.g., "Sigmoid").
     * @param useNormalization If true, prepends a normalization layer based on dataset stats.
     */
    void createNeuralNetwork(const std::vector<int>& hiddenSizes, 
                            const std::string& hiddenAct = "ReLU", 
                            const std::string& outputAct = "Sigmoid",
                            bool useNormalization = false);

    /**
     * @brief Automatically constructs a network architecture based on a high-level project type.
     * @param type The project goal (e.g., CLASSIFICATION, FORECASTING).
     * @param hiddenSizes Vector containing the number of neurons for each hidden layer.
     * @param windowSize Number of steps to look back (for Forecasting/SlidingWindow).
     */
    void createNeuralNetwork(owProjectType type, const std::vector<int>& hiddenSizes, int windowSize = 5);

    /**
     * @brief Triggers the training process using the configured optimizer.
     * 
     * If the optimizer supports global optimization (like L-BFGS), it will take 
     * control of the entire network. Otherwise, it runs the standard backpropagation loop.
     */
    void train();

    /**
     * @brief Executes a standard iterative training loop with backpropagation.
     */
	void runStandardTrainingLoop();

    /**
     * @brief Sets a time limit for the training process.
     * @param seconds Maximum training time in seconds (0.0 for no limit).
     */
    void setMaxTrainingTime(double seconds) { m_maxTime = seconds; }

    /**
     * @brief Gets the configured maximum training time.
     * @return Time in seconds.
     */
    double getMaxTrainingTime() const { return m_maxTime; }

    /**
     * @brief Sets the target error threshold for early stopping.
     * @param error Minimum loss value to reach.
     */
    void setMinimumError(float error) { m_minError = error; }

    /**
     * @brief Sets the minimum MAPE (percentage error) threshold for early stopping.
     * @param mape Minimum percentage error (e.g., 0.5 for 0.5%). Set to 0.0 to disable.
     */
    void setMinimumPercentageError(float mape) { m_minMape = mape; }

    /**
     * @brief Gets the minimum error threshold.
     * @return Float loss value.
     */
    float getMinimumError() const { return m_minError; }

    /**
     * @brief Gets the minimum MAPE threshold.
     * @return Percentage value.
     */
    float getMinimumPercentageError() const { return m_minMape; }
    
    /**
     * @brief Sets the tolerance for detecting loss stagnation.
     * @param tol Minimum change in loss to be considered an improvement.
     */
    void setLossStagnationTolerance(float tol) { m_lossStagnationTolerance = tol; }

    /**
     * @brief Gets the loss stagnation tolerance.
     * @return Float tolerance value.
     */
    float getLossStagnationTolerance() const { return m_lossStagnationTolerance; }

    /**
     * @brief Sets whether loss stagnation control is enabled.
     * @param enabled True to enable, false to disable.
     */
    void setLossStagnationEnabled(bool enabled) { m_lossStagnationEnabled = enabled; }

    /**
     * @brief Checks if loss stagnation control is enabled.
     * @return True if enabled, false otherwise.
     */
    bool isLossStagnationEnabled() const { return m_lossStagnationEnabled; }

    /**
     * @brief Sets the number of epochs to wait before stopping due to stagnation.
     * @param epochs Patience count.
     */
    void setLossStagnationPatience(int epochs) { m_lossStagnationPatience = epochs; }

    /**
     * @brief Gets the loss stagnation patience.
     * @return Integer epoch count.
     */
    int getLossStagnationPatience() const { return m_lossStagnationPatience; }
    
    /**
     * @brief Sets the reason why the last training session finished.
     * @param reason Description string.
     */
    void setTrainingFinishReason(const std::string& reason) { m_finishReason = reason; }

    /**
     * @brief Gets the reason for the last training termination (e.g., "Epoch Limit", "Convergence").
     * @return Reason string.
     */
    std::string getTrainingFinishReason() const { return m_finishReason; }

    /**
     * @brief Internal method to record actual training time.
     * @param time Time in seconds.
     */
    void setActualTrainingTime(double time) { m_actualTrainingTime = time; }

    /**
     * @brief Records the number of epochs actually completed.
     * @param num Epoch count.
     */
    void setTrainingEpochNum(int num) { m_actualEpochs = num; }

    /**
     * @brief Returns the number of epochs completed during the last training.
     * @return Integer epoch count.
     */
    int getTrainingEpochNum() const { return m_actualEpochs; }

    /**
     * @brief Returns the total time spent in the last training session.
     * @return Time in seconds.
     */
    double getTrainingTime() const { return m_actualTrainingTime; }

    /**
     * @brief Sets the loss value from the last training epoch.
     * @param err Float loss value.
     */
    void setLastTrainError(float err) { m_lastTrainLoss = err; }

    /**
     * @brief Returns the training loss from the most recent epoch.
     * @return Float loss value.
     */
    float getLastTrainError() const { return m_lastTrainLoss; }

    /**
     * @brief Sets the validation loss from the last training epoch.
     * @param err Float loss value.
     */
    void setLastValError(float err) { m_lastValLoss = err; }

    /**
     * @brief Returns the validation loss from the most recent epoch.
     * @return Float loss value.
     */
    float getLastValError() const { return m_lastValLoss; }

    /**
     * @brief Evaluates the network's performance on a given dataset.
     * @param input Tensor containing input features.
     * @param target Tensor containing ground truth values.
     * @param tolerance Precision tolerance for accuracy calculation (default 5%).
     * @return An EvaluationReport containing various metrics.
     */
    EvaluationReport evaluatePerformance(const owTensor<float, 2>& input, const owTensor<float, 2>& target, float tolerance = 0.05f);

    /**
     * @brief Evaluates the network's performance on the internal dataset's test set.
     * @param tolerance Precision tolerance for accuracy calculation (default 0.5f).
     * @return An EvaluationReport containing various metrics.
     */
    EvaluationReport evaluatePerformance(float tolerance = 0.5f);

    /**
     * @brief Performs a forward pass and returns the human-readable label for the prediction.
     * Useful for classification tasks with categorical targets.
     * @param input Input tensor [1, InputFeatures].
     * @param targetVarIdx Which target variable to use (default 0).
     * @return Label string.
     */
    std::string predictLabel(const owTensor<float, 2>& input, int targetVarIdx = 0);

    /**
     * @brief Prints the current training status in a standardized format.
     * @param epoch Current epoch number.
     * @param trainLoss Current training loss.
     * @param valLoss Current validation loss (0.0 if not available).
     * @param elapsedTime Time elapsed since the start of training.
     */
    void printTrainingStatus(int epoch, float trainLoss, float valLoss, double elapsedTime);

    /**
     * @brief Prints the contents of an EvaluationReport to standard output.
     * @param report The report to print.
     */
    void printEvaluationReport(const EvaluationReport& report) const;

    /**
     * @brief Sets the frequency of status prints during training.
     * @param interval Epoch interval (e.g., 10 means print every 10 epochs).
     */
    void setPrintEpochInterval(int interval) { m_printInterval = std::max(1, interval); }

    /**
     * @brief Gets the current print interval.
     * @return Integer interval.
     */
    int getPrintEpochInterval() const { return m_printInterval; }

    /**
     * @brief Enables or disables console output during training.
     * @param enable True to enable printing.
     */
    void setEnablePrinting(bool enable) { m_enablePrinting = enable; }

    /**
     * @brief Configures regularization for all layers in the network.
     * @param type Regularization type (0: None, 1: L1, 2: L2).
     */
    void setRegularization(int type);

    /**
     * @brief Adds a custom layer to the network architecture.
     * @param layer Shared pointer to an owLayer implementation.
     */
    void addLayer(std::shared_ptr<owLayer> layer);

    /**
     * @brief Performs a forward pass through all layers.
     * @param input Input tensor (BatchSize x InputDim).
     * @return Output tensor from the last layer.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input);

    /**
     * @brief High-level prediction method that applies inverse normalization if needed.
     * @param input Input features.
     * @return Processed output (e.g., actual price in forecasting).
     */
    owTensor<float, 2> predict(const owTensor<float, 2>& input);

    /**
     * @brief Prediction using the last sample from the dataset.
     */
    owTensor<float, 2> predict();

    /**
     * @brief Performs a backward pass (backpropagation) through all layers.
     * @param prediction The output from the forward pass.
     * @param target The expected ground truth values.
     */
    void backward(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target);

    /**
     * @brief Computes the loss value given prediction and target.
     */
    float calculateLoss(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target);

    /**
     * @brief Updates internal parameters for all layers based on gradients.
     */
    void trainStep();

    /**
     * @brief Resets any internal state (useful for recurrent or stateful networks).
     */
    void reset();

    /**
     * @brief Performs recursive multi-step forecasting starting from the last sample in the dataset.
     * @param steps Number of future steps to predict.
     * @return Tensor containing the sequence of future predictions [steps, TargetSize].
     */
    owTensor<float, 2> forecast(int steps = 1);

    /**
     * @brief Performs recursive multi-step forecasting starting from a specific sample.
     * 
     * Starting from an initial sample, this method predicts the next step, 
     * feeds that prediction back as input, and repeats for the specified number of steps.
     * Useful for long-term time-series projections.
     * 
     * @param initialSample The starting observation [1, InputFeatures].
     * @param steps Number of future steps to predict.
     * @return Tensor containing the sequence of future predictions [steps, TargetSize].
     */
    owTensor<float, 2> forecast(const owTensor<float, 2>& initialSample, int steps);

    /**
     * @brief Assigns an optimizer to the network.
     * @param opt Shared pointer to an owOptimizer (e.g., ADAM, SGD).
     */
    void setOptimizer(std::shared_ptr<owOptimizer> opt);

    /**
     * @brief Assigns a loss function to the network.
     * @param loss Shared pointer to an owLoss (e.g., MSE, CrossEntropy).
     */
    void setLoss(std::shared_ptr<owLoss> loss);

    /**
     * @brief Returns a shared pointer to the current loss function.
     */
    std::shared_ptr<owLoss> getLoss() { return m_loss; }

    /**
     * @brief Calculates the total number of trainable parameters (Weights + Biases).
     * @return Total count.
     */
    size_t getTotalParameterCount() const;

    /**
     * @brief Flattens all network parameters into a single 1D tensor.
     * 
     * Used primarily by global optimizers like L-BFGS or Conjugate Gradient.
     * @param target The 1D tensor to fill.
     */
    void getGlobalParameters(owTensor<float, 1>& target) const;

    /**
     * @brief Restores network parameters from a flattened 1D tensor.
     * @param source The 1D tensor containing the parameters.
     */
    void setGlobalParameters(const owTensor<float, 1>& source);

    /**
     * @brief Flattens all network gradients into a single 1D tensor.
     * @param target The 1D tensor to fill.
     */
    void getGlobalGradients(owTensor<float, 1>& target) const;

    /**
     * @brief Returns a shared pointer to the current optimizer. 
     * If no optimizer is set, initializes a default ADAM optimizer.
     * @return Shared pointer to owOptimizer.
     */
    std::shared_ptr<owOptimizer> getOptimizer();

    /**
     * @brief Returns a vector of shared pointers to all layers in the network.
     * @return Vector of shared pointers to owLayer.
     */
    std::vector<std::shared_ptr<owLayer>> getLayers();

    /**
     * @brief Retrieves the names of all layers for display or logging.
     */
    owTensor<std::string, 1> getLayerNames() const;

    /**
     * @brief Retrieves the number of neurons in each layer.
     */
    owTensor<float, 1> getNeuronNums() const;

    /**
     * @brief Manually sets the project type to influence high-level behaviors like prediction scaling.
     */
    void setProjectType(owProjectType type) { m_projectType = type; }

    /**
     * @brief Retrieves the current project type configuration.
     */
    owProjectType getProjectType() const { return m_projectType; }

    /**
     * @brief Indicates if the model has been trained or updated via partial fitting.
     */
    bool isPartiallyFitted() const { return m_isPartiallyFitted; }

    /**
     * @brief Performs training on a specific set of data for a fixed number of iterations.
     * Useful for online learning or fine-tuning.
     */
    void partialFit(const owTensor<float, 2>& input, const owTensor<float, 2>& target, int steps = 1);

    /**
     * @brief Toggles training mode for the entire network.
     * 
     * Propagates the training status to all layers. This is essential for 
     * stateful layers (CacheLayer, Dropout) that behave differently 
     * during training vs. validation or inference.
     * 
     * @param training True to enable training mode, false for evaluation.
     */
    void setTraining(bool training);

    /**
     * @brief Saves the current model architecture and weights to an XML file.
     */
    bool saveToXML(const std::string& filename);

    /**
     * @brief Loads the model architecture and weights from an XML file.
     */
    bool loadFromXML(const std::string& filename);

    /**
     * @brief Retrieves input normalization statistics (min and max) from the dataset.
     * @param min Output tensor to store minimum values [1, InputDim].
     * @param max Output tensor to store maximum values [1, InputDim].
     */
    void getInputMinMax(owTensor<float, 2>& min, owTensor<float, 2>& max) const;

    /**
     * @brief Retrieves target normalization statistics (min and max) from the dataset.
     * @param min Output tensor to store minimum values [1, TargetDim].
     * @param max Output tensor to store maximum values [1, TargetDim].
     */
    void getTargetMinMax(owTensor<float, 2>& min, owTensor<float, 2>& max) const;

private:
    std::vector<std::shared_ptr<owLayer>> m_layers; ///< Ordered list of layers.
    std::shared_ptr<owOptimizer> m_optimizer;      ///< Strategy for updating weights.
    std::shared_ptr<owLoss> m_loss;                ///< Criterion for measuring error.
    std::shared_ptr<owDataset> m_dataset;          ///< Data source for training.
    owProjectType m_projectType = owProjectType::CUSTOM; ///< Project task categorization.
    
    int m_maxEpochs = 1000;
    double m_maxTime = 0.0;
    float m_minError = 0.001f;
    float m_minMape = 0.0f;
    float m_lossStagnationTolerance = 0.0005f;
    int m_lossStagnationPatience = 50;
    bool m_lossStagnationEnabled = true;
    
    int m_actualEpochs = 0;
    double m_actualTrainingTime = 0.0;
    float m_lastTrainLoss = 0.0f;
    float m_lastValLoss = 0.0f;
    std::string m_finishReason = "Maximum Epoch Num";
    bool m_isPartiallyFitted = false;
    
    int m_printInterval = 5;
    bool m_enablePrinting = true;
    int m_regType = 2; // Default L2

    std::mt19937 m_rng; ///< High-quality random number generator.

    /**
     * @brief Factory method to create activation functions by name.
     */
    std::shared_ptr<owActivation> createActivationByName(const std::string& name);
    };


} // namespace ow
