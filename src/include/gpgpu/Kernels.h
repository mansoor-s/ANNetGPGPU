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

#ifndef SWIG
#include "../containers/TrainingSet.h"

#include "../gpgpu/2DArray.h"
#include "../gpgpu/SOMExport.h"

#include "../math/Functions.h"

#include <cassert>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#endif


/*
 * BP kernels
 */
std::vector<float>
hostBPCalcDelta(const thrust::device_vector<float> &vNeurOut,
		const std::vector<float> &vTrainOut );

std::vector<thrust::device_vector<float> >
hostBPPropagateFW(const std::vector<ANNGPGPU::F2DArray> &vEdgeMatrices,
		const std::vector<ANNGPGPU::F2DArray> &vBiasEdgeMatrices,
		const std::vector<float> &vInput,
		const ANN::TransfFunction &function);

void
hostBPPropagateBW(std::vector<ANNGPGPU::F2DArray> &dvEdgeMatricesI,
		std::vector<ANNGPGPU::F2DArray> &dvMomentums,
		std::vector<thrust::device_vector<float> > &vErrorDeltas,
		const std::vector<thrust::device_vector<float> > &vNeuronValues,
		const float &fLearningRate,
		const float &fWeightDecay,
		const float &fMomentum,
		const ANN::TransfFunction &function);

/*
 * SOM kernels
 */
//////////////////////////////////////////////////////////////////////////////////////////////
float hostGetMax(const thrust::device_vector<float>& vec, unsigned int &ID);
float hostGetMin(const thrust::device_vector<float>& vec, unsigned int &ID);

//////////////////////////////////////////////////////////////////////////////////////////////
ANNGPGPU::BMUExport
hostSOMFindBMNeuronID(std::vector<ANNGPGPU::SplittedNetExport*> &SExp,
		const float &fConscienceRate);

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename BinaryFunction>
void
hostSOMPropagateBW(std::vector<ANNGPGPU::SplittedNetExport*> &SExp,
		const ANNGPGPU::BMUExport &,
		const float &fSigmaT,
		const float &fLearningRate,
		const BinaryFunction &binaryDistFunc  );

void
hostSOMTraining( std::vector<ANNGPGPU::SplittedNetExport*> &SExp,
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles,
		const float &fSigma0,
		const float &fLearningRate0,
		const float &fConscienceRate,
		float (*pfnDecay)(const float &, const float &, const float &),
		const ANN::DistFunction &pDistFunc );

#endif /* ANKERNELS_H_ */
