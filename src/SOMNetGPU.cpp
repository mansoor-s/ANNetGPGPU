/*
 * ANSOMNetGPU.cpp
 *
 *  Created on: 01.04.2012
 *      Author: dgrat
 */

#include "include/gpgpu/SOMNetGPU.h"
#include "include/math/Functions.h"
#include "include/SOMLayer.h"
#include "include/base/AbsNeuron.h"
#include <cuda.h>
#include <gpgpu/helper_cuda.h>
#include <gpgpu/timer.h>

namespace ANNGPGPU {

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
////////////////////////////////////////////////////////////////////////////////

int SOMNetGPU::GetCudaDeviceCount() const {
	int GPU_N = 0;
	printf("Check for CUDA-capable devices\n");
	checkCudaErrors(cudaGetDeviceCount(&GPU_N) );
	if (GPU_N > MAX_GPU_COUNT) {
	    GPU_N = MAX_GPU_COUNT;
	}
	printf("CUDA-capable device count: %i\n", GPU_N);

	return GPU_N;
}

std::vector<SplittedNetExport*> SOMNetGPU::SplitDeviceData() const {
	unsigned int iStart 		= 0;
	unsigned int iStop 		= 0;
	unsigned int iSizeOfLayer 	= GetOPLayer()->GetNeurons().size();
	unsigned int iDeviceCount 	= 1;//GetCudaDeviceCount();
	// To make things easy ..
	if(iSizeOfLayer%iDeviceCount != 0) {
		iDeviceCount = 1;
	}
	
	std::vector<SplittedNetExport*> vRes(iDeviceCount);
	printf("Computing with %d GPUs ..\n", iDeviceCount);
	for(int i = 0; i < iDeviceCount; i++) { 
		checkCudaErrors(cudaSetDevice(i) );

		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;

		// Copy conscience information
		thrust::host_vector<float> hvConscience(iStop-iStart+1);
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			hvConscience[j] = m_pOPLayer->GetNeuron(j+iStart)->GetValue();
		}

		printf(".. Copy edges: %d/%d\n", i+1, iDeviceCount);
		F2DArray f2dEdges 	= GetOPLayer()->ExpEdgesIn(iStart, iStop);
		printf(".. Copy positions: %d/%d\n", i+1, iDeviceCount);
		F2DArray f2dPositions 	= GetOPLayer()->ExpPositions(iStart, iStop);

		// Create network export container
		vRes[i] = new SplittedNetExport(f2dEdges, f2dPositions, hvConscience);
	}
	return vRes;
}

void SOMNetGPU::CombineDeviceData(std::vector<SplittedNetExport*> &SExp) {
	unsigned int iStart 		= 0;
	unsigned int iStop 		= 0;
	unsigned int iSizeOfLayer 	= GetOPLayer()->GetNeurons().size();
	unsigned int iDeviceCount 	= 1;//GetCudaDeviceCount();
	// To make things easy ..
	if(iSizeOfLayer%iDeviceCount != 0) {
		iDeviceCount = 1;
	}

	for(int i = 0; i < iDeviceCount; i++) {
		checkCudaErrors(cudaSetDevice(i) );

		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;

		// Copy back conscience
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			m_pOPLayer->GetNeuron(j+iStart)->SetValue((*SExp.at(i)->dvConscience)[j]);
		}
		
		printf(".. Copy back edges: %d/%d\n", i+1, iDeviceCount);
		// Copy weights between neurons of the input and output layer
		GetOPLayer()->ImpEdgesIn(SExp.at(i)->f2dEdges, iStart, iStop);
		
		// delete old network export container
		delete SExp.at(i);
	}
	// delete old network export container
	SExp.clear();
}

SOMNetGPU::SOMNetGPU() {
	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 		= NULL;

	m_iCycle 		= 0;
	m_fSigma0 		= 0.f;
	m_fSigmaT 		= 0.f;
	m_fLearningRate 	= 0.5f;

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;

	// Conscience mechanism
	m_fConscienceRate 	= 0.f;
	
	// mexican hat shaped function for this SOM
	SetDistFunction(&ANN::Functions::fcn_gaussian);

	m_fTypeFlag 	= ANN::ANNetSOM;
}

SOMNetGPU::SOMNetGPU(AbsNet *pNet) {
	if(pNet == NULL)
		return;

	std::vector<unsigned int> vDimI = ((ANN::SOMLayer*)(pNet->GetIPLayer() ))->GetDim();
	std::vector<unsigned int> vDimO = ((ANN::SOMLayer*)(pNet->GetOPLayer() ))->GetDim();

	// Copy weights between neurons of the input and output layer
	ANN::F2DArray f2dEdges = pNet->GetOPLayer()->ExpEdgesIn();
	// Copy positions of the neurons in the output layer
	ANN::F2DArray f2dPosistions = pNet->GetOPLayer()->ExpPositions();
	// Create the net finally
	CreateSOM(vDimI, vDimO, f2dEdges, f2dPosistions);
	// Copy training set
	SetTrainingSet(pNet->GetTrainingSet() );

	m_fTypeFlag 	= ANN::ANNetSOM;
}

SOMNetGPU::~SOMNetGPU() {
	checkCudaErrors(cudaDeviceReset() );
}

void SOMNetGPU::Training(const unsigned int &iCycles) {
	assert(iCycles > 0);
	assert(m_fSigma0 > 0.f);
	if(GetTrainingSet() == NULL) {
		std::cout<<"No training set available!"<<std::endl;
		return;
	}

	printf("Copy memory from host to device ..\n");
	std::vector<SplittedNetExport*> SExp = SplitDeviceData();

	StartTimer();

	printf("Calculate SOM ..\n");
	hostSOMTraining(SExp,
		*GetTrainingSet(),
		iCycles,
		m_fSigma0,
		m_fLearningRate,
		m_fConscienceRate,
		&ANN::fcn_decay,
		*GetDistFunction() );
	
	printf("GPU Processing time: %f (ms)\n", GetTimer() );

	// Write edge matrix back
	std::cout<<"Copy memory from device to host .."<<std::endl;
	// Copy data from device to host
	CombineDeviceData(SExp);	
	std::cout<<".. Finished"<<std::endl;
}

}
