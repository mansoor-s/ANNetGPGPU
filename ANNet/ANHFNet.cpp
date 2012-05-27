/*
 * HFNet.cpp
 *
 *  Created on: 21.02.2011
 *      Author: dgrat
 */

#include <cassert>

#include <basic/ANEdge.h>

#include <ANHFNeuron.h>
#include <ANHFNet.h>
#include <ANHFLayer.h>

#include <math/ANFunctions.h>
#include <containers/ANTrainingSet.h>

using namespace ANN;


HFNet::HFNet() {
	m_fTypeFlag 	= ANNetHopfield;
}

HFNet::HFNet(const unsigned int &iW, const unsigned int &iH) {
	Resize(iW, iH);

	m_fTypeFlag 	= ANNetHopfield;
}

HFNet::HFNet(AbsNet *pNet) : AbsNet(pNet) {
	assert( pNet != NULL );

	m_fTypeFlag 	= ANNetHopfield;
}

HFNet::~HFNet() {

}

void HFNet::Resize(const unsigned int &iW, const unsigned int &iH) {
	EraseAll();

	m_iWidth 	= iW;
	m_iHeight 	= iH;

	HFLayer *pIOLayer = new HFLayer(iW, iH);
	AddLayer(pIOLayer);

	m_pIPLayer = pIOLayer;
	m_pOPLayer = pIOLayer;

	pIOLayer->ConnectLayer(true);
}

void HFNet::PropagateFW() {
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>( m_pIPLayer->GetNeurons().size() ); i++) {
		m_pIPLayer->GetNeuron(i)->CalcValue();
	}
}

void HFNet::CalculateMatrix() {
	int iLength 	= (m_iHeight * m_iWidth); // == m_pIPLayer->GetNeurons().size()
	int iMatSize 	= iLength * iLength;
	float *pMat 	= new float[iMatSize];
	memset(pMat, 0, sizeof(float) * iMatSize);

	// Calculate weight matrix
	#pragma omp parallel for
	for(int Y = 0; Y < iLength; Y++) {		// run through every src neuron
		#pragma omp parallel for
		for(int X = 0; X < iLength; X++) {	// run through every dst neuron
			if(Y == X)	{					// skip neuron
				//pMat[Y*iLength+X] = 0.f;
				continue;
			}
			else {
				float fSum = 0.f;
				for(unsigned int i = 0; i < m_pTrainingData->GetNrElements(); i++) {
					fSum += m_pTrainingData->GetInput(i)[X] * m_pTrainingData->GetInput(i)[Y];
				}
				pMat[X*iLength+Y] = fSum;
				pMat[Y*iLength+X] = fSum;
			}
		}
	}

	((HFLayer *)m_pIPLayer)->ClearWeights();
	// Apply matrix
	((HFLayer*)m_pIPLayer)->ConnectLayer(pMat, true);

	// free memory
	delete [] pMat;

}

void HFNet::PropagateBW() {
	CalculateMatrix();
}

void HFNet::SetInput(float *pInputArray) {
	// SetInput(inpu, size, layer)
	AbsNet::SetInput(pInputArray, m_pIPLayer->GetNeurons().size(), 0);
}

void HFNet::SetInput(std::vector<float> vInputArray) {
	// SetInput(inpu, layer)
	AbsNet::SetInput(vInputArray, 0);
}

