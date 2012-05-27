/*
 * ANKernels.h
 *
 *  Created on: 29.03.2012
 *      Author: dgrat
 */

#ifndef ANKERNELS_H_
#define ANKERNELS_H_

#include <cassert>
#include <containers/ANTrainingSet.h>
#include <thrust/device_vector.h>
#include <gpgpu/ANMatrix.h>


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
hostSOMPropagateBW( ANN::Matrix &SOMEdgeMatrix,
		const ANN::Matrix &SOMPositionMatrix,
		const thrust::device_vector<float> &dvInputVector,
		const unsigned int BMUID,
		const float &fSigmaT,
		const float &fLearningRate );

void
hostSOMTraining( ANN::Matrix &SOMEdgeMatrix,
		const ANN::Matrix &SOMPositionMatrix,
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles,
		const float &fSigma0,
		const float &fLearningRate0,
		float (*pfnDecay)(const float &, const float &, const float &) );

#endif /* ANKERNELS_H_ */
