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
#include <ctime>
#include <gpgpu/helper_cuda.h>

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
	
	std::vector<SplittedNetExport*> vRes;
	
	printf("Computing with %d GPUs ..\n", iDeviceCount);
	#pragma omp parallel for
	for(int i = 0; i < iDeviceCount; i++) { 
		checkCudaErrors(cudaSetDevice(i) );

		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;

		// Copy conscience information
		thrust::host_vector<float> hvConscience(iStop-iStart+1);
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			hvConscience[j] = m_pOPLayer->GetNeuron(j+iStart)->GetValue();
		}

		SplittedNetExport *pExp = new SplittedNetExport(GetOPLayer()->ExpEdgesIn(iStart, iStop), 	// Copy weights between neurons of the input and output layer
								GetOPLayer()->ExpPositions(iStart, iStop), 	// Copy positions of the neurons in the output layer
								hvConscience);
		vRes.push_back(pExp);
	}
	return vRes;
}

void SOMNetGPU::CombineDeviceData(const std::vector<SplittedNetExport*> &SExp) {
	unsigned int iStart 		= 0;
	unsigned int iStop 		= 0;
	unsigned int iSizeOfLayer 	= GetOPLayer()->GetNeurons().size();
	unsigned int iDeviceCount 	= 1;//GetCudaDeviceCount();

	#pragma omp parallel for
	for(int i = 0; i < iDeviceCount; i++) {
		checkCudaErrors(cudaSetDevice(i) );

		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;

		// Copy weights between neurons of the input and output layer
		GetOPLayer()->ImpEdgesIn(SExp.at(i)->f2dEdges, iStart, iStop);

		// Copy back conscience
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			m_pOPLayer->GetNeuron(j+iStart)->SetValue((*SExp.at(i)->dvConscience)[j]);
		}
		delete SExp.at(i);
	}
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

}

void SOMNetGPU::Training(const unsigned int &iCycles) {
	assert(iCycles > 0);
	assert(m_fSigma0 > 0.f);
	if(GetTrainingSet() == NULL) {
		std::cout<<"No training set available!"<<std::endl;
		return;
	}
	
	clock_t begin = clock();

	printf("Copy memory from host to device ..\n");
	std::vector<SplittedNetExport*> SExp = SplitDeviceData();

	printf("Calculate SOM ..\n");
	hostSOMTraining(SExp,
		*GetTrainingSet(),
		iCycles,
		m_fSigma0,
		m_fLearningRate,
		m_fConscienceRate,
		&ANN::fcn_decay,
		*GetDistFunction() );
	
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	std::cout<<"Training cycles finished properly after "<<elapsed_secs<<" s"<<std::endl;
	// Write edge matrix back
	std::cout<<"Copy memory from device to host .."<<std::endl;
	// Copy data from device to host
	CombineDeviceData(SExp);	
	std::cout<<".. Finished"<<std::endl;
}

}
