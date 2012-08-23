/*
 * ANBPNetGPU.cpp
 *
 *  Created on: 14.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <ANAbsNeuron.h>
#include <ANBPNet.h>
#include <ANBPNeuron.h>
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

float BPNetGPU::SetOutput(const std::vector<float> &vOutArray) {
	assert( m_pOPLayer != NULL );
	assert( vOutArray.size() == m_pOPLayer->GetNeurons().size() );

	PropagateFW();

	std::vector<float> vOutDelta = hostBPCalcDelta(m_vNeuronVals.back(), vOutArray);

	float fError = 0.f;
	for(int i = 0; i < vOutDelta.size(); i++) {
		fError += pow(vOutDelta[i], 2) / 2.f;
		// TODO very slow
		m_pOPLayer->GetNeuron(i)->SetErrorDelta(vOutDelta[i]);
	}

	return fError;
}

void BPNetGPU::GetErrorDeltas() {
	m_vOutDeltas.clear();
	m_dvOutDeltas.clear();
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		std::vector<float> vOutDelta;
		for(unsigned int j = 0; j < m_lLayers.at(i)->GetNeurons().size(); j++) {
			vOutDelta.push_back(m_lLayers.at(i)->GetNeuron(j)->GetErrorDelta() );
		}
		m_vOutDeltas.push_back(vOutDelta);
		m_dvOutDeltas.push_back(thrust::device_vector<float>(vOutDelta.begin(), vOutDelta.end()) );
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
		m_dvOutDeltas,//m_vOutDeltas,					// const
		m_vNeuronVals,					// const
		GetLearningRate()				// const
	);
}

std::vector<float> BPNetGPU::TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress) {
	// Retrive weight matrices
	GetEdgeMatrices();
//	GetErrorDeltas();
	// Train the network and calc new weight matrices
	std::vector<float> vRes = ANN::BPNet::TrainFromData(iCycles, fTolerance, bBreak, fProgress);
	// Update the weights
	RefreshEdges();
	//RefreshNeurons();

	// Return error deltas
	return vRes;
}

}
