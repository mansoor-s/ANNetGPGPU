/*
 * ANSOMNetGPU.cpp
 *
 *  Created on: 01.04.2012
 *      Author: dgrat
 */

#include <gpgpu/ANSOMNetGPU.h>
#include <math/ANFunctions.h>
#include <ANSOMLayer.h>
#include <basic/ANAbsNeuron.h>
#include <cuda.h>


namespace ANN {

int SOMNetGPU::GetCudaDeviceCount() {
	int iCount = 0;

	if(cudaGetDeviceCount(&iCount) != cudaSuccess)
		return 0;

	std::cout<<iCount<<" cuda-capable device(s) found."<<std::endl;
	return iCount;
}

void SOMNetGPU::SplitDeviceData(const int &iDeviceCount) {
	for(unsigned int i = 0; i < iDeviceCount; i++) {
		if(cudaSetDevice(i) != cudaSuccess) {
			return;
		}
		else {
			// TODO SPLIT
		}
	}
}

SOMNetGPU::SOMNetGPU() {
	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 	= NULL;

	m_iCycle 		= 0;
	m_fSigma0 		= 0.f;
	m_fSigmaT 		= 0.f;
	m_fLearningRate = 0.5f;

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;

	// Conscience mechanism
	m_fConscienceRate 	= 0.f;
	
	// mexican hat shaped function for this SOM
	SetDistFunction(&Functions::fcn_gaussian);

	m_fTypeFlag 	= ANNetSOM;

	GetCudaDeviceCount();
}

SOMNetGPU::SOMNetGPU(AbsNet *pNet) {
	if(pNet == NULL)
		return;

	std::vector<unsigned int> vDimI = ((SOMLayer*)(pNet->GetIPLayer() ))->GetDim();
	std::vector<unsigned int> vDimO = ((SOMLayer*)(pNet->GetOPLayer() ))->GetDim();

	// Copy weights between neurons of the input and output layer
	ANN::F2DArray f2dEdges = pNet->GetOPLayer()->ExpEdgesIn();
	// Copy positions of the neurons in the output layer
	ANN::F2DArray f2dPosistions = pNet->GetOPLayer()->ExpPositions();
	// Create the net finally
	CreateSOM(vDimI, vDimO, f2dEdges, f2dPosistions);
	// Copy training set
	SetTrainingSet(pNet->GetTrainingSet() );

	m_fTypeFlag 	= ANNetSOM;

	GetCudaDeviceCount();
}

SOMNetGPU::~SOMNetGPU() {

}

void SOMNetGPU::Training(const unsigned int &iCycles) {
	assert(iCycles > 0);
	assert(m_fSigma0 > 0.f);
	if(GetTrainingSet() == NULL) {
		std::cout<<"No training set available!"<<std::endl;
		return;
	}

	m_EdgeMat = m_pOPLayer->ExpEdgesIn();
	m_PosiMat = ((SOMLayer*)m_pOPLayer)->ExpPositions();

	unsigned int iSize = m_pOPLayer->GetNeurons().size();
	thrust::host_vector<float> hvConscience(iSize);
	thrust::device_vector<float> dvConscience;
	for(unsigned int i = 0; i < iSize; i++) {
		hvConscience[i] = m_pOPLayer->GetNeuron(i)->GetValue();
	}
	dvConscience = hvConscience;
	
	std::cout<< "Process the SOM now" <<std::endl;
	hostSOMTraining(dvConscience,
			m_EdgeMat,
			m_PosiMat,
			*GetTrainingSet(),
			iCycles,
			m_fSigma0,
			m_fLearningRate,
			m_fConscienceRate,
			&ANN::fcn_decay);

	std::cout<<"Training cycles finished properly"<<std::endl;
	// Write edge matrix back
	std::cout<<"Copy device memory back .."<<std::endl;
	m_pOPLayer->ImpEdgesIn(m_EdgeMat);
	
	for(unsigned int i = 0; i < iSize; i++) {
		m_pOPLayer->GetNeuron(i)->SetValue(dvConscience[i]);
	}
	
	std::cout<<".. Finished"<<std::endl;
}

}
