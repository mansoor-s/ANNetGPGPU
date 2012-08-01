/*
 * ANBPNetGPU.cpp
 *
 *  Created on: 14.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <ANAbsNeuron.h>
#include <ANBPNet.h>
#include <ANBPLayer.h>
#include <math/ANFunctions.h>
#include <gpgpu/ANBPNetGPU.h>

namespace ANN {

BPNetGPU::BPNetGPU() {

}

BPNetGPU::~BPNetGPU() {
	// TODO Auto-generated destructor stub
}

float BPNetGPU::SetOutput(const std::vector<float> &vOutArray) {
	assert( m_pOPLayer != NULL );
	assert( vOutArray.size() == m_pOPLayer->GetNeurons().size() );

	PropagateFW();
	m_vOutDelta = hostBPCalcDelta(m_vNeuronVals.back(), vOutArray);

	float fError = 0.f;
	for(int i = 0; i < m_vOutDelta.size(); i++) {
		fError += pow(m_vOutDelta[i], 2) / 2.f;
	}

	return fError;
}

float BPNetGPU::SetOutput(const std::vector<float> &outputArray, const unsigned int &layerID) {
	assert( layerID < m_lLayers.size() );
	assert( outputArray.size() == m_lLayers[layerID]->GetNeurons().size() );

	PropagateFW();
	m_vOutDelta = hostBPCalcDelta(m_vNeuronVals.at(layerID), outputArray);

	float fError = 0.f;
	for(int i = 0; i < m_vOutDelta.size(); i++) {
		fError += pow(m_vOutDelta[i], 2) / 2.f;
	}

	return fError;
}

float BPNetGPU::SetOutput(float *pOutArray, const unsigned int &size, const unsigned int &layerID) {
	assert( layerID < m_lLayers.size() );
	assert( size == m_lLayers[layerID]->GetNeurons().size() );

	std::vector<float> vOutArray;
	std::copy ( pOutArray, pOutArray+size, vOutArray.begin() );

	PropagateFW();
	m_vOutDelta = hostBPCalcDelta(m_vNeuronVals.at(layerID), vOutArray);

	float fError = 0.f;
	for(int i = 0; i < m_vOutDelta.size(); i++) {
		fError += pow(m_vOutDelta[i], 2) / 2.f;
	}

	return fError;
}

/*
void BPNetGPU::ExpNeurVals() {
	m_vNeuronVals.clear();
	for(int i = 0; i < GetLayers().size(); i++) {
		thrust::host_vector<float> hvLayer;
		for(int j = 0; j < GetLayer(i)->GetNeurons().size(); j++) {
			hvLayer.push_back(GetLayer(i)->GetNeuron(j)->GetValue() );
		}
		m_vNeuronVals.push_back(hvLayer);
	}
}
*/

void BPNetGPU::PropagateFW() {
	/*
	 * Copy edges in matrix
	 */
	m_vEdgeMatrices.clear();
	m_vBiasEdgeMatrices.clear();
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		ANN::BPLayer *pLayer = (ANN::BPLayer *)m_lLayers.at(i);
		m_vEdgeMatrices.push_back(pLayer->ExpEdgesOut() );
		m_vBiasEdgeMatrices.push_back(pLayer->ExpBiasEdgesOut() );
	}

	/*
	 * Process
	 */
	std::vector<float> vInput;
	for(unsigned int i = 0; i < GetIPLayer()->GetNeurons().size(); i++) {
		ANN::AbsNeuron *pNeuron = GetIPLayer()->GetNeuron(i);
		vInput.push_back(pNeuron->GetValue() );
	}

	m_vNeuronVals =	hostBPPropagateFW (
		m_vEdgeMatrices,				// const
		m_vBiasEdgeMatrices,			// const
		vInput,							// const
		GetTransfFunction()->normal		// fptr
	);
}

void BPNetGPU::PropagateBW() {
	/*
	 * Copy edges in matrix
	 */
	m_vEdgeMatrices.clear();
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		ANN::BPLayer *pLayer = (ANN::BPLayer *)m_lLayers.at(i);
		m_vEdgeMatrices.push_back(pLayer->ExpEdgesOut() );
	}

	/*
	 * Process
	 */
	m_vEdgeMomentumMatrices = hostBPPropagateBW	(
		m_vEdgeMatrices,				// -> changed due to training
		m_vNeuronVals,					// const
		m_vOutDelta,					// const
		GetLearningRate(),				// const
		GetTransfFunction()->derivate	// fptr
	);
}

std::vector<float> BPNetGPU::TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress) {
	std::vector<float> vOutput;
	for(unsigned int i = 0; i < GetOPLayer()->GetNeurons().size(); i++) {
		ANN::AbsNeuron *pNeuron = GetOPLayer()->GetNeuron(i);
		vOutput.push_back(pNeuron->GetValue() );
	}
}

}
