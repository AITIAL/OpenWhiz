/*
 * owLBFGSOptimizer.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owOptimizer.hpp"
#include <deque>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace ow {

class owNeuralNetwork;
class owDataset;

/**
 * @class owLBFGSOptimizer
 * @brief Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimizer.
 *
 * L-BFGS is a Quasi-Newton method that approximates the Hessian matrix (second-order information) 
 * using a limited history of gradient updates. It is a **Global Optimizer**, meaning it operates 
 * on all parameters of the network simultaneously.
 *
 * @par Comparison: L-BFGS vs. BFGS
 * - **Memory Efficiency:** L-BFGS uses @f$ O(mN) @f$ memory (where @f$ m @f$ is history size), 
 *   making it suitable for large-scale problems. Standard BFGS uses @f$ O(N^2) @f$, which 
 *   becomes prohibitive for large networks.
 * - **Approximation Strategy:** While BFGS maintains a full @f$ N \times N @f$ matrix, L-BFGS 
 *   reconstructs the curvature information using a "two-loop recursion" over the recent history.
 * - **When to use:**
 *     - Use **L-BFGS** for large neural networks where memory is a constraint.
 *     - Use **BFGS** (Standard) for small to medium networks where higher precision and 
 *       faster convergence to deep minima are required.
 *
 * **Advantages:**
 * - **Unmatched Precision:** Often finds the global or local minimum with extremely high accuracy.
 * - **Iteration Efficiency:** Requires far fewer iterations (epochs) than SGD or Adam.
 * - **Automatic Step-Size:** Uses an internal "Line Search" to find the optimal learning rate.
 *
 * @note **Industrial/Computer Use:** Highly recommended for high-precision modeling where 
 * accuracy is paramount and RAM is sufficient.
 * @note **Mobile/Web Use:** Generally NOT recommended due to the high RAM usage.
 * 
 * @important **GPU Compatibility:** L-BFGS is currently only supported in CPU mode. 
 */
 class owLBFGSOptimizer : public owOptimizer {
private:
    size_t m_m = 100; ///< Increased history size for better Hessian approximation.
    std::deque<std::vector<double>> s_list; ///< Difference in parameters (history).
    std::deque<std::vector<double>> y_list; ///< Difference in gradients (history).
    std::deque<double> rho_list; ///< Precomputed scalar coefficients.

    /**
     * @brief Computes the dot product of two vectors.
     * @param a First vector.
     * @param b Second vector.
     * @return double The resulting dot product.
     */
    inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0;
        size_t n = a.size();
        for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

public:
    /**
     * @brief Constructs an L-BFGS optimizer.
     * @param lr The initial step size factor (default: 1.0).
     * @param m The size of the update history (default: 100).
     */
    explicit owLBFGSOptimizer(float lr = 1.0f, int m = 100) : m_m(m) {
        this->m_learningRate = lr;
    }

    /**
     * @brief Performs global optimization on the entire neural network.
     *
     * Implements the L-BFGS algorithm with backtracking line search.
     * This method bypasses layer-wise updates and optimizes the whole network
     * as a single vector.
     *
     * @param nn The neural network instance.
     * @param ds The dataset for training.
     */
    void optimizeGlobal(owNeuralNetwork* nn, owDataset* ds) override {
        size_t nParams = nn->getTotalParameterCount();
        if (nParams == 0) return;

        auto trainIn = ds->getTrainInput();
        auto trainTarget = ds->getTrainTarget();

        owTensor<float, 1> x_f(nParams), g_f(nParams);
        nn->getGlobalParameters(x_f);

        std::vector<double> x(nParams), g(nParams), d(nParams), x_next(nParams);
        for(size_t i=0; i<nParams; ++i) x[i] = (double)x_f.data()[i];

        bool firstPass = true;
        auto compute_f_g = [&](const std::vector<double>& cur_x, std::vector<double>& cur_g) {
            for(size_t i=0; i<nParams; ++i) x_f.data()[i] = (float)cur_x[i];
            nn->setGlobalParameters(x_f);
            nn->reset(); // Clear state (like sliding window history) for consistent evaluation
            auto pred = nn->forward(trainIn);
            
            // After the very first forward pass, we signal all layers to lock their cache.
            // This is essential for owCacheLayer to stop recording and switch to playback.
            if (firstPass) {
                for (auto& layer : nn->getLayers()) layer->lockCache();
                firstPass = false;
            }

            const auto& activeTarget = nn->getActiveTarget(trainTarget);
            float loss = nn->calculateLoss(pred, activeTarget);
            nn->reset(); // Clear state again before backward to ensure it uses fresh forward path
            nn->forward(trainIn); // Re-run forward to populate state for backward pass
            nn->backward(pred, activeTarget);
            nn->getGlobalGradients(g_f);
            for(size_t i=0; i<nParams; ++i) cur_g[i] = (double)g_f.data()[i];
            return (double)loss;
        };

        // Ensure all layers have access to the target tensor for local loss/caching
        for (auto& layer : nn->getLayers()) layer->setTarget(&trainTarget);

        double f = compute_f_g(x, g);
        double bestLoss = f;
        int patience = 0;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int k = 1; k <= nn->getMaximumEpochNum(); ++k) {
            // 1. Direction Calculation
            if (s_list.empty()) {
                for(size_t i=0; i<nParams; ++i) d[i] = -g[i];
            } else {
                std::vector<double> q = g;
                std::vector<double> alphas(s_list.size());
                for (int i = (int)s_list.size() - 1; i >= 0; --i) {
                    alphas[i] = rho_list[i] * dot(s_list[i], q);
                    for(size_t j=0; j<nParams; ++j) q[j] -= alphas[i] * y_list[i][j];
                }

                double sy = dot(s_list.back(), y_list.back());
                double yy = dot(y_list.back(), y_list.back());
                double gamma = (std::abs(yy) > 1e-18) ? sy / yy : 1.0;

                for(size_t i=0; i<nParams; ++i) d[i] = q[i] * gamma;
                for (size_t i = 0; i < s_list.size(); ++i) {
                    double beta = rho_list[i] * dot(y_list[i], d);
                    for(size_t j=0; j<nParams; ++j) d[j] += s_list[i][j] * (alphas[i] - beta);
                }
                for(size_t i=0; i<nParams; ++i) d[i] *= -1.0;
            }

            // 2. Line Search (Strong Wolfe)
            double g_norm = 0;
            for(size_t i=0; i<nParams; ++i) g_norm += g[i]*g[i];
            g_norm = std::sqrt(g_norm);

            double g_dot_d = dot(g, d);
            // If direction is not descent, reset history
            if (g_dot_d > -1e-9 * g_norm) {
                s_list.clear(); y_list.clear(); rho_list.clear();
                for(size_t i=0; i<nParams; ++i) d[i] = -g[i];
                g_dot_d = dot(g, d);
            }

            double step = 1.0;
            if (g_norm > 1.0) step = std::min(1.0, 1.0 / g_norm);
            bool success = false;
            std::vector<double> g_next(nParams);
            double f_next = f;

            const double c1 = 1e-4; // Armijo constant
            const double c2 = 0.9;  // Wolfe constant

            for (int i = 0; i < 60; ++i) {
                for(size_t j=0; j<nParams; ++j) x_next[j] = x[j] + step * d[j];
                f_next = compute_f_g(x_next, g_next);

                // Armijo condition
                if (f_next <= f + c1 * step * g_dot_d + 1e-12) {
                    double g_next_dot_d = dot(g_next, d);
                    // Strong Wolfe Curvature condition
                    if (std::abs(g_next_dot_d) <= c2 * std::abs(g_dot_d)) {
                        success = true;
                        break;
                    }
                    if (step < 1e-12) { success = true; break; }
                }
                step *= 0.6; 
                if (step < 1e-18) break;
            }

            if (!success && s_list.size() > 0) {
                // If line search failed with approximated Hessian, reset history and retry
                s_list.clear(); y_list.clear(); rho_list.clear();
                continue;
            }

            if (!success) {
                nn->setTrainingFinishReason("Minimum Precision Limit");
                break;
            }

            // 3. Update History
            std::vector<double> sk(nParams), yk(nParams);
            for(size_t j=0; j<nParams; ++j) {
                sk[j] = x_next[j] - x[j];
                yk[j] = g_next[j] - g[j];
            }
            double ys = dot(yk, sk);
            double yy = dot(yk, yk);
            
            if (ys > 1e-14 * yy && ys > 1e-18) {
                if (s_list.size() >= m_m) { s_list.pop_front(); y_list.pop_front(); rho_list.pop_front(); }
                s_list.push_back(sk); y_list.push_back(yk); rho_list.push_back(1.0 / ys);
            }

            x = x_next; g = g_next; f = f_next;
            nn->setLastTrainError((float)f);

            // Validation loss calculation (optional but consistent)
            float valLoss = 0.0f;
            auto valIn = ds->getValInput();
            if (valIn.size() > 0) {
                auto valTarget = ds->getValTarget();
                auto valPred = nn->forward(valIn);
                valLoss = nn->calculateLoss(valPred, valTarget);
                nn->setLastValError(valLoss);
            }

            // --- MAPE Based Stopping ---
            if (nn->getMinimumPercentageError() > 0.0f) {
                float currentMape = 0.0f;
                // Compute current predictions to get MAPE
                auto pred = nn->forward(trainIn);
                const auto& activeTarget = nn->getActiveTarget(trainTarget);
                size_t n = pred.shape()[0], outDim = pred.shape()[1];
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < outDim; ++j) {
                        float p = pred(i, j), t = activeTarget(i, j);
                        if (std::abs(t) > 1e-7f) currentMape += std::abs((p - t) / t);
                    }
                }
                currentMape = (currentMape / (n * outDim)) * 100.0f;
                if (currentMape <= nn->getMinimumPercentageError()) {
                    nn->setTrainingFinishReason("Minimum Error");
                    nn->setTrainingEpochNum(k);
                    if (nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                        auto now = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> currentElapsed = now - startTime;
                        nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                    }
                    break;
                }
            }

            if (nn->getPrintEpochInterval() > 0 && (k == 1 || k % nn->getPrintEpochInterval() == 0)) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> currentElapsed = now - startTime;
                nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
            }

            // Force at least 100 epochs if it's early and accuracy isn't perfect yet
            bool canStop = (k > 100) && nn->isLossStagnationEnabled();

            if (f < bestLoss - (double)nn->getLossStagnationTolerance()) {
                bestLoss = f; patience = 0;
            } else {
                patience++;
            }

            if ((canStop && patience >= nn->getLossStagnationPatience()) || f < nn->getMinimumError()) {
                nn->setTrainingFinishReason(f < nn->getMinimumError() ? "Minimum Error" : "Loss Stagnation");
                nn->setTrainingEpochNum(k);
                // Print final epoch if not already printed
                if (nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                }
                break;
            }
            
            nn->setTrainingEpochNum(k);
            nn->setTrainingFinishReason("Maximum Epoch Num");

            // Print final epoch if limit reached and not already printed
            if (k == nn->getMaximumEpochNum() && nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> currentElapsed = now - startTime;
                nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
            }
        }
        for(size_t i=0; i<nParams; ++i) x_f.data()[i] = (float)x[i];
        nn->setGlobalParameters(x_f);
    }

    /**
     * @brief Layer-wise update is not used for L-BFGS.
     * This method is an empty override as L-BFGS uses optimizeGlobal().
     */
    void update(owTensor<float, 2>&, const owTensor<float, 2>&) override {}

    /**
     * @brief Returns the name of the optimizer.
     */
    std::string getOptimizerName() const override { return "L-BFGS"; }

    /**
     * @brief Creates a deep copy of the L-BFGS optimizer.
     * @return std::shared_ptr<owOptimizer> A shared pointer to the cloned instance.
     */
    std::shared_ptr<owOptimizer> clone() const override { return std::make_shared<owLBFGSOptimizer>(m_learningRate); }

    /**
     * @brief Indicates support for global optimization.
     * @return true Always returns true for L-BFGS.
     */
    bool supportsGlobalOptimization() const override { return true; }
};

} // namespace ow
