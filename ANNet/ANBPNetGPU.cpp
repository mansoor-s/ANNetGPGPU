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
	m_fTypeFlag 		= ANNetBP;
	SetTransfFunction(&ANN::Functions::fcn_log);	// TODO not nice
}

BPNetGPU::~BPNetGPU() {
	// TODO Auto-generated destructor stub
}
/*
float BPNetGPU::SetOutput(const std::vector<float> &vOutArray) {
	assert( m_pOPLayer != NULL );
	assert( vOutArray.size() == m_pOPLayer->GetNeurons().size() );

	PropagateFW();

	m_vOutDeltas.clear();
	for(unsigned int i = 0; i < m_lLayers.size()-1; i++) {
		std::vector<float> vOutDelta;
		for(unsigned int j = 0; j < m_lLayers.at(i)->GetNeurons().size(); j++) {
			vOutDelta.push_back(m_lLayers.at(i)->GetNeuron(j)->GetErrorDelta() );
		}
		m_vOutDeltas.push_back(vOutDelta);
	}
	std::vector<float> vOutDelta = hostBPCalcDelta(m_vNeuronVals.back(), vOutArray);
	m_vOutDeltas.push_back(vOutDelta);

	float fError = 0.f;
	for(int i = 0; i < m_vOutDeltas.back().size(); i++) {
		fError += pow(m_vOutDeltas.back()[i], 2) / 2.f;
	}

	return fError;
}

float BPNetGPU::SetOutput(const std::vector<float> &outputArray, const unsigned int &layerID) {
	assert( layerID < m_lLayers.size() );
	assert( outputArray.size() == m_lLayers[layerID]->GetNeurons().size() );

	PropagateFW();

	m_vOutDeltas.clear();
	for(unsigned int i = 0; i < m_lLayers.size()-1; i++) {
		std::vector<float> vOutDelta;
		for(unsigned int j = 0; j < m_lLayers.at(i)->GetNeurons().size(); j++) {
			vOutDelta.push_back(m_lLayers.at(i)->GetNeuron(j)->GetErrorDelta() );
		}
		m_vOutDeltas.push_back(vOutDelta);
	}
	std::vector<float> vOutDelta = hostBPCalcDelta(m_vNeuronVals.at(layerID), outputArray);
	m_vOutDeltas.push_back(vOutDelta);

	float fError = 0.f;
	for(int i = 0; i < m_vOutDeltas.back().size(); i++) {
		fError += pow(m_vOutDeltas.back()[i], 2) / 2.f;
	}

	return fError;
}

float BPNetGPU::SetOutput(float *pOutArray, const unsigned int &size, const unsigned int &layerID) {
	assert( layerID < m_lLayers.size() );
	assert( size == m_lLayers[layerID]->GetNeurons().size() );

	std::vector<float> vOutArray;
	std::copy ( pOutArray, pOutArray+size, vOutArray.begin() );

	PropagateFW();

	m_vOutDeltas.clear();
	for(unsigned int i = 0; i < m_lLayers.size()-1; i++) {
		std::vector<float> vOutDelta;
		for(unsigned int j = 0; j < m_lLayers.at(i)->GetNeurons().size(); j++) {
			vOutDelta.push_back(m_lLayers.at(i)->GetNeuron(j)->GetErrorDelta() );
		}
		m_vOutDeltas.push_back(vOutDelta);
	}
	std::vector<float> vOutDelta = hostBPCalcDelta(m_vNeuronVals.at(layerID), vOutArray);
	m_vOutDeltas.push_back(vOutDelta);

	float fError = 0.f;
	for(int i = 0; i < m_vOutDeltas.back().size(); i++) {
		fError += pow(m_vOutDeltas.back()[i], 2) / 2.f;
	}

	return fError;
}
*/

void BPNetGPU::GetErrorDeltas() {
	m_vOutDeltas.clear();
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		std::vector<float> vOutDelta;
		for(unsigned int j = 0; j < m_lLayers.at(i)->GetNeurons().size(); j++) {
			vOutDelta.push_back(m_lLayers.at(i)->GetNeuron(j)->GetErrorDelta() );
		}
		m_vOutDeltas.push_back(vOutDelta);
	}
}

void BPNetGPU::RefreshNeurons() {
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		std::vector<float> vNeurVals(m_vNeuronVals.at(i).size() );
		thrust::copy(m_vNeuronVals.at(i).begin(), m_vNeuronVals.at(i).end(), vNeurVals.begin());

		for(unsigned int j = 0; j < m_lLayers.at(i)->GetNeurons().size(); j++) {
			m_lLayers.at(i)->GetNeuron(j)->SetValue(vNeurVals.at(j) );
		}
	}
}

void BPNetGPU::RefreshErrorDeltas() {
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		std::vector<float> vNeurVals(m_vNeuronVals.at(i).size() );
		thrust::copy(m_vNeuronVals.at(i).begin(), m_vNeuronVals.at(i).end(), vNeurVals.begin());

		for(unsigned int j = 0; j < m_lLayers.at(i)->GetNeurons().size(); j++) {
			m_lLayers.at(i)->GetNeuron(j)->SetErrorDelta(m_vOutDeltas.at(i).at(j) );
		}
	}
}

void BPNetGPU::RefreshEdges() {
	// Use m_vEdgeMatricesI for import,
	// because only that one got changed in "PropagateBW()"
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		ANN::BPLayer *pLayer = (ANN::BPLayer *)m_lLayers.at(i);
		if(!(pLayer->GetFlag() & ANLayerInput) ) {
			pLayer->ImpEdgesIn(m_vEdgeMatricesI.at(i-1) );
		}
	}
}

std::vector<float> BPNetGPU::GetCurrentInput() {
	std::vector<float> vInput;
	for(unsigned int i = 0; i < GetIPLayer()->GetNeurons().size(); i++) {
		ANN::AbsNeuron *pNeuron = GetIPLayer()->GetNeuron(i);
		vInput.push_back(pNeuron->GetValue() );
	}
	return vInput;
}

void BPNetGPU::GetEdgeMatrices() {
	// Copy edges in matrix
	m_vEdgeMatricesO.clear();
	m_vEdgeMatricesI.clear();
	m_vBiasEdgeMatrices.clear();
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		ANN::BPLayer *pLayer = (ANN::BPLayer *)m_lLayers.at(i);
		if(!(pLayer->GetFlag() & ANLayerOutput) ) {
			m_vEdgeMatricesO.push_back(pLayer->ExpEdgesOut() );
		}
		if(!(pLayer->GetFlag() & ANLayerInput) ) {
			m_vEdgeMatricesI.push_back(pLayer->ExpEdgesIn() );
			if(pLayer->GetBiasNeuron() != NULL) {
				m_vBiasEdgeMatrices.push_back(pLayer->ExpBiasEdgesOut() );
			}
		}
	}
}

void BPNetGPU::PropagateFW() {
	m_vNeuronVals =	hostBPPropagateFW (
		m_vEdgeMatricesI,				// const
		m_vBiasEdgeMatrices,			// const
		GetCurrentInput()				// const
	);

	/*
	 * Write neurons back
	 * TODO optimize, is very slow
	 */
	RefreshNeurons();
}

void BPNetGPU::PropagateBW() {
	GetErrorDeltas();

	/*
	 * Process
	 */
	m_vEdgeMomentumMatrices = hostBPPropagateBW	(
		m_vEdgeMatricesO,				// -> changed due to training
		m_vEdgeMatricesI,
		m_vOutDeltas,					// const
		m_vNeuronVals,					// const
		GetLearningRate()				// const
	);

	/*
	 * Write edges back
	 * TODO optimize, is very slow
	 */
	RefreshErrorDeltas();
}

std::vector<float> BPNetGPU::TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress) {
	// Retrive weight matrices
	GetEdgeMatrices();
	// Train the network and calc new weight matrices
	std::vector<float> vRes = ANN::BPNet::TrainFromData(iCycles, fTolerance, bBreak, fProgress);
	// Update the weights
	RefreshEdges();

	// Return error deltas
	return vRes;
}

}
