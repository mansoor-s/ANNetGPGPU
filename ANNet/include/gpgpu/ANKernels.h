/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/

#ifndef ANKERNELS_H_
#define ANKERNELS_H_

#include <cassert>
#include <vector>
#include <containers/ANTrainingSet.h>
#include <thrust/device_vector.h>
#include <gpgpu/ANMatrix.h>
#include <math/ANFunctions.h>

/*
 * BP kernels
 */
std::vector<float>
hostBPCalcDelta(
		const thrust::device_vector<float> &vNeurOut,
		const std::vector<float> &vTrainOut );

std::vector<thrust::device_vector<float> >
hostBPPropagateFW(
		const std::vector<ANN::Matrix> &vEdgeMatrices,
		const std::vector<ANN::Matrix> &vBiasEdgeMatrices,
		const std::vector<float> &vInput);

void
hostBPPropagateBW(
		std::vector<ANN::Matrix> &dvEdgeMatricesI,
		std::vector<ANN::Matrix> &dvMomentums,
		std::vector<thrust::device_vector<float> > &vErrorDeltas,
		const std::vector<thrust::device_vector<float> > &vNeuronValues,
		const float &fLearningRate,
		const float &fWeightDecay,
		const float &fMomentum);

/*
 * SOM kernels
 */
//////////////////////////////////////////////////////////////////////////////////////////////
float hostGetMax(const thrust::device_vector<float>& vec, unsigned int &ID);
float hostGetMin(const thrust::device_vector<float>& vec, unsigned int &ID);

//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int
hostSOMFindBMNeuronID(
		const ANN::Matrix &SOMEdgeMatrix,
		const thrust::device_vector<float> &InputVector );

//////////////////////////////////////////////////////////////////////////////////////////////
void
hostSOMPropagateBW(
		ANN::Matrix &SOMEdgeMatrix,
		const ANN::Matrix &SOMPositionMatrix,
		const thrust::device_vector<float> &dvInputVector,
		const unsigned int BMUID,
		const float &fSigmaT,
		const float &fLearningRate );

void
hostSOMTraining(
		ANN::Matrix &SOMEdgeMatrix,
		const ANN::Matrix &SOMPositionMatrix,
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles,
		const float &fSigma0,
		const float &fLearningRate0,
		float (*pfnDecay)(const float &, const float &, const float &) );

#endif /* ANKERNELS_H_ */
