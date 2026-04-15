/****************************************************************************
 * Copyright (c) 2025 AITIAL LOGICIEL SAS, Paris, France                    *
 *                                                                          *
 * Permission is hereby granted, free of charge, to any person obtaining a  *
 * copy of this software and associated documentation files (the            *
 * "Software"), to deal in the Software without restriction, including      *
 * without limitation the rights to use, copy, modify, merge, publish,      *
 * distribute, distribute with modifications, sublicense, and/or sell       *
 * copies of the Software, and to permit persons to whom the Software is    *
 * furnished to do so, subject to the following conditions:                 *
 *                                                                          *
 * The above copyright notice and this permission notice should not be      *
 * deleted from the source form of the Software.                            *
 *                                                                          *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS  *
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF               *
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.   *
 * IN NO EVENT SHALL THE ABOVE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,   *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR    *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR    *
 * THE USE OR OTHER DEALINGS IN THE SOFTWARE.                               *
 *                                                                          *
 * Except as contained in this notice, the name(s) of the above copyright   *
 * holders shall not be used in advertising or otherwise to promote the     *
 * sale, use or other dealings in this Software without prior written       *
 * authorization.                                                           *
 ****************************************************************************/


#pragma once

// Core Tensor engine
#include "core/owTensor.hpp"

// Utilities, Maths, Data
#include "data/owDataset.hpp"
#include "data/owStatistics.hpp"

// Forward declarations and base classes
#include "core/owNeuralNetwork.hpp"
#include "nonlinearities/owActivation.hpp"
#include "losses/owLoss.hpp"
#include "optimizers/owOptimizer.hpp"
#include "layers/owLayer.hpp"

// Activations
#include "nonlinearities/owIdentityActivation.hpp"
#include "nonlinearities/owLeakyReLUActivation.hpp"
#include "nonlinearities/owReLUActivation.hpp"
#include "nonlinearities/owSigmoidActivation.hpp"
#include "nonlinearities/owTanhActivation.hpp"

// Optimizers (Full implementations)
#include "optimizers/owADAMOptimizer.hpp"
#include "optimizers/owConjugateGradientOptimizer.hpp"
#include "optimizers/owLBFGSOptimizer.hpp"
#include "optimizers/owMomentumOptimizer.hpp"
#include "optimizers/owRMSPropOptimizer.hpp"
#include "optimizers/owSGDOptimizer.hpp"

// Losses (Full implementations)
#include "losses/owBinaryCrossEntropyLoss.hpp"
#include "losses/owCategoricalCrossEntropyLoss.hpp"
#include "losses/owHuberLoss.hpp"
#include "losses/owMarginRankingLoss.hpp"
#include "losses/owMeanAbsoluteErrorLoss.hpp"
#include "losses/owMeanSquaredErrorLoss.hpp"
#include "losses/owPinballLoss.hpp"
#include "losses/owWeightedMeanSquaredErrorLoss.hpp"

// Layers (Full implementations - depends on Optimizer/Loss/Activation)
#include "layers/owAdditionLayer.hpp"
#include "layers/owAffineLayer.hpp"
#include "layers/owAnomalyDetectionLayer.hpp"
#include "layers/owAttentionLayer.hpp"
#include "layers/owBoundingLayer.hpp"
#include "layers/owClippingLayer.hpp"
#include "layers/owClusterLayer.hpp"
#include "layers/owConcatenateLayer.hpp"
#include "layers/owDateTimeEncodingLayer.hpp"
#include "layers/owInverseNormalizationLayer.hpp"
#include "layers/owLinearLayer.hpp"
#include "layers/owLSTMLayer.hpp"
#include "layers/owMultiHeadAttentionLayer.hpp"
#include "layers/owNormalizationLayer.hpp"
#include "layers/owPositionEncodingLayer.hpp"
#include "layers/owPrincipalComponentAnalysisLayer.hpp"
#include "layers/owProbabilityLayer.hpp"
#include "layers/owProjectionLayer.hpp"
#include "layers/owDistanceLayer.hpp"
#include "layers/owQuantileLayer.hpp"
#include "layers/owRankingLayer.hpp"
#include "layers/owSmoothingLayer.hpp"
#include "layers/owTrendLayer.hpp"
#include "layers/owChangeRateLayer.hpp"
#include "layers/owRescalingLayer.hpp"
#include "layers/owSequentialLayer.hpp"
#include "layers/owSlidingWindowLayer.hpp"

// Implementation of owNeuralNetwork inline methods (must be last)
#include "core/owNeuralNetwork.inl"
