/*
 * ANBPNetGPU.cpp
 *
 *  Created on: 14.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <gpgpu/ANBPNetGPU.h>
#include <ANBPNet.h>
#include <ANBPLayer.h>

namespace ANN {

BPNetGPU::BPNetGPU() {
	// TODO Auto-generated constructor stub

}

BPNetGPU::~BPNetGPU() {
	// TODO Auto-generated destructor stub
}

void BPNetGPU::PropagateFW() {

}

void BPNetGPU::PropagateBW() {
	/*
	 * Copy edges in matrix
	 */
	m_vEdgeMatrices.clear();
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		ANN::BPLayer *pLayer = (ANN::BPLayer *)m_lLayers.at(i);
		m_vEdgeMatrices.push_back(pLayer->ExpEdgesOut());
	}
}

std::vector<float> BPNetGPU::TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress) {

}

}
