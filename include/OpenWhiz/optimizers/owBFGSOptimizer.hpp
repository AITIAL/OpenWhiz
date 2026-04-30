/*
 * owBFGSOptimizer.hpp
 *
 *  Created on: Apr 27, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owOptimizer.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace ow {

class owNeuralNetwork;
class owDataset;

/**
 * @class owBFGSOptimizer
 * @brief Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer.
 *
 * BFGS is a Quasi-Newton method that approximates the full Hessian matrix (second-order information).
 * Unlike its limited-memory variant (L-BFGS), standard BFGS stores and updates the complete N x N 
 * inverse Hessian approximation matrix.
 *
 * @par Mathematical Foundation
 * Newton's method uses the direction @f$ d = -H^{-1} g @f$, where @f$ H @f$ is the Hessian.
 * BFGS avoids calculating the actual Hessian by iteratively updating an approximation of the 
 * inverse Hessian @f$ B \approx H^{-1} @f$.
 *
 * @par Comparison: BFGS vs. L-BFGS
 * - **Memory Usage:** BFGS requires @f$ O(N^2) @f$ memory to store the full matrix, while L-BFGS 
 *   requires @f$ O(mN) @f$ where @f$ m @f$ is history size.
 * - **Precision:** BFGS is generally more stable and provides higher precision for small to 
 *   medium-sized networks (e.g., < 5,000 parameters) as it maintains full curvature information.
 * - **Convergence:** For small models, BFGS often converges in fewer iterations and reaches 
 *   deeper minima than L-BFGS.
 * - **When to use:** 
 *     - Use **BFGS** for high-precision tasks with small neural networks (e.g., time-series forecasting 
 *       with a few layers).
 *     - Use **L-BFGS** for large networks or when system RAM is limited.
 *
 * @note **Accuracy:** Small models (like CAC40 example) often converge much faster and to 
 * deeper minima with Full BFGS compared to L-BFGS.
 */
class owBFGSOptimizer : public owOptimizer {
private:
    /**
     * @brief Computes the dot product of two vectors.
     * @param a First vector.
     * @param b Second vector.
     * @return The scalar dot product @f$ \sum a_i b_i @f$.
     */
    inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0;
        size_t n = a.size();
        for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

public:
    /**
     * @brief Constructs a BFGS optimizer.
     * @param lr The initial learning rate/step size factor (default: 1.0).
     */
    explicit owBFGSOptimizer(float lr = 1.0f) {
        this->m_learningRate = lr;
    }

    /**
     * @brief Performs global optimization on the entire neural network using the BFGS algorithm.
     *
     * The optimization follows these core steps:
     * 1. **Search Direction:** Computes @f$ d_k = -B_k \nabla f(x_k) @f$ where @f$ B_k @f$ is the 
     *    approximate inverse Hessian.
     * 2. **Line Search:** Finds a step size @f$ \alpha @f$ satisfying Armijo and Wolfe conditions.
     * 3. **Update:** Calculates @f$ s_k = x_{k+1} - x_k @f$ and @f$ y_k = g_{k+1} - g_k @f$.
     * 4. **Hessian Update:** Updates the matrix @f$ B_{k+1} @f$ using the BFGS formula.
     *
     * @param nn Pointer to the neural network to be optimized.
     * @param ds Pointer to the dataset containing training and validation data.
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

        // Inverse Hessian Approximation Matrix (Initialized as Identity Matrix I)
        // Memory complexity: O(N^2)
        std::vector<double> invH(nParams * nParams, 0.0);
        auto resetHessian = [&]() {
            std::fill(invH.begin(), invH.end(), 0.0);
            for(size_t i=0; i<nParams; ++i) invH[i * nParams + i] = 1.0;
        };
        resetHessian();

        bool firstPass = true;
        /**
         * @brief Internal helper to compute loss and gradients at a given parameter point.
         * @param cur_x The current parameter vector.
         * @param cur_g Output vector where gradients will be stored.
         * @return The calculated loss (float converted to double).
         */
        auto compute_f_g = [&](const std::vector<double>& cur_x, std::vector<double>& cur_g) {
            for(size_t i=0; i<nParams; ++i) x_f.data()[i] = (float)cur_x[i];
            nn->setGlobalParameters(x_f);
            nn->reset();
            auto pred = nn->forward(trainIn);
            
            if (firstPass) {
                for (auto& layer : nn->getLayers()) layer->lockCache();
                firstPass = false;
            }

            const auto& activeTarget = nn->getActiveTarget(trainTarget);
            float loss = nn->calculateLoss(pred, activeTarget);
            nn->reset();
            nn->forward(trainIn);
            nn->backward(pred, activeTarget);
            nn->getGlobalGradients(g_f);
            for(size_t i=0; i<nParams; ++i) cur_g[i] = (double)g_f.data()[i];
            return (double)loss;
        };

        for (auto& layer : nn->getLayers()) layer->setTarget(&trainTarget);

        double f = compute_f_g(x, g);
        double bestLoss = f;
        int patience = 0;
        double lastStep = 1.0; // Memory for step size
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int k = 1; k <= nn->getMaximumEpochNum(); ++k) {
            // 1. Direction Calculation: d = -H * g
            // Standard matrix-vector multiplication for O(N^2) complexity.
            for (size_t i = 0; i < nParams; ++i) {
                double sum = 0;
                for (size_t j = 0; j < nParams; ++j) {
                    sum += invH[i * nParams + j] * g[j];
                }
                d[i] = -sum;
            }

            // 2. Descent Direction Check
            // A direction is descent if g.d < 0. If it isn't, curvature info is stale.
            double g_dot_d = dot(g, d);
            if (g_dot_d >= 0) {
                resetHessian();
                for (size_t i = 0; i < nParams; ++i) d[i] = -g[i];
                g_dot_d = dot(g, d);
            }

            // 3. Sophisticated Line Search with Memory
            // Implements backtracking line search with Armijo condition.
            double step = (k == 1) ? 1.0 / std::sqrt(dot(g, g) + 1e-10) : lastStep;
            if (step > 1.0) step = 1.0;
            if (step < 1e-6) step = 1e-6;

            bool success = false;
            std::vector<double> g_next(nParams);
            double f_next = f;

            const double c1 = 1e-4; // Sufficient decrease constant
            const double c2 = 0.01; // Curvature constant

            for (int i = 0; i < 150; ++i) {
                for(size_t j=0; j<nParams; ++j) x_next[j] = x[j] + step * d[j];
                f_next = compute_f_g(x_next, g_next);

                // Armijo condition check: f(x + step*d) <= f(x) + c1 * step * (g^T d)
                if (f_next < f + c1 * step * g_dot_d + 1e-9) {
                    double g_next_dot_d = dot(g_next, d);
                    // Curvature condition (Wolfe): g(x + step*d)^T d >= c2 * g(x)^T d
                    if (g_next_dot_d >= c2 * g_dot_d) {
                        success = true;
                        lastStep = step * 1.2; 
                        if (lastStep > 1.0) lastStep = 1.0;
                        break;
                    }
                }
                step *= 0.5; 
                if (step < 1e-40) break;
            }

            if (!success) {
                // Machine Epsilon Last Resort: Attempt a tiny perturbation to break numerical stalls.
                double eps = std::numeric_limits<double>::epsilon();
                bool perturbed = false;
                for (size_t i = 0; i < nParams; ++i) {
                    if (std::abs(g[i]) > 1e-35) {
                        x_next[i] = x[i] - (g[i] > 0 ? eps : -eps);
                        perturbed = true;
                    } else {
                        x_next[i] = x[i];
                    }
                }
                if (perturbed) {
                    f_next = compute_f_g(x_next, g_next);
                    if (f_next < f) {
                        success = true;
                        lastStep = 1e-6;
                    }
                }
            }

            if (!success) {
                resetHessian();
                lastStep = 1.0;
                if (patience > 10) {
                    nn->setTrainingFinishReason("Minimum Precision Limit");
                    break; 
                }
                patience++;
                continue;
            }

            // 4. BFGS Matrix Update
            // sk = difference in parameters, yk = difference in gradients
            std::vector<double> sk(nParams), yk(nParams);
            for(size_t j=0; j<nParams; ++j) {
                sk[j] = x_next[j] - x[j];
                yk[j] = g_next[j] - g[j];
            }

            double ys = dot(yk, sk);
            
            // Self-scaling on first step or reset: Improves initial Hessian approximation.
            if (k == 1 || g_dot_d >= 0) {
                double yy = dot(yk, yk);
                if (ys > 1e-30 && yy > 1e-30) {
                    double scale = ys / yy;
                    for (size_t i = 0; i < nParams; ++i) {
                        for (size_t j = 0; j < nParams; ++j) {
                            if (i == j) invH[i * nParams + j] = scale;
                            else invH[i * nParams + j] = 0.0;
                        }
                    }
                }
            }

            // Compute H * yk for the update formula
            std::vector<double> Hs(nParams, 0.0);
            for(size_t i=0; i<nParams; ++i) {
                for(size_t j=0; j<nParams; ++j) Hs[i] += invH[i * nParams + j] * yk[j];
            }
            double sHs = dot(yk, Hs);

            // BFGS Update Formula for Inverse Hessian
            // H_new = (I - sk*yk'/yk'*sk) * H_old * (I - yk*sk'/yk'*sk) + sk*sk'/yk'*sk
            if (ys > 1e-30 && sHs > 1e-30) {
                std::vector<double> bfgs_v(nParams);
                for (size_t i = 0; i < nParams; ++i) bfgs_v[i] = sk[i] / ys - Hs[i] / sHs;

                for (size_t i = 0; i < nParams; ++i) {
                    for (size_t j = 0; j < nParams; ++j) {
                        double update = (sk[i] * sk[j]) / ys 
                                      - (Hs[i] * Hs[j]) / sHs 
                                      + (bfgs_v[i] * bfgs_v[j]) * sHs;
                        invH[i * nParams + j] += update;
                    }
                }
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

            // Print Status
            if (nn->getPrintEpochInterval() > 0 && (k == 1 || k % nn->getPrintEpochInterval() == 0)) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> currentElapsed = now - startTime;
                nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
            }

            // Stopping Criteria
            if (f < nn->getMinimumError()) {
                nn->setTrainingFinishReason("Minimum Error");
                nn->setTrainingEpochNum(k);
                break;
            }

            if (f < bestLoss - (double)nn->getLossStagnationTolerance()) {
                bestLoss = f; patience = 0;
            } else {
                patience++;
            }

            if (patience >= nn->getLossStagnationPatience()) {
                nn->setTrainingFinishReason("Loss Stagnation");
                nn->setTrainingEpochNum(k);
                break;
            }
            
            nn->setTrainingEpochNum(k);
            nn->setTrainingFinishReason("Maximum Epoch Num");
        }
        for(size_t i=0; i<nParams; ++i) x_f.data()[i] = (float)x[i];
        nn->setGlobalParameters(x_f);
    }

    void update(owTensor<float, 2>&, const owTensor<float, 2>&) override {}
    std::string getOptimizerName() const override { return "BFGS"; }
    std::shared_ptr<owOptimizer> clone() const override { return std::make_shared<owBFGSOptimizer>(m_learningRate); }
    bool supportsGlobalOptimization() const override { return true; }
};

} // namespace ow
