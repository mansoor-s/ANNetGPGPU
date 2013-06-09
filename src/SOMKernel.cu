#ifndef _SOMKERNELS_
#define _SOMKERNELS_

#include "include/math/Random.h"
#include "include/gpgpu/Kernels.h"
#include "include/gpgpu/Functors.h"
#include "include/gpgpu/helper_cuda.h"

#include <cfloat>
#include <cassert>
#include <cmath>

//#include <omp.h>

using namespace ANNGPGPU;


float hostGetMax(const thrust::device_vector<float>& vec, unsigned int &ID) {
	// create implicit index sequence [0, 1, 2, ... ]
	thrust::counting_iterator<unsigned int> begin(0);
	thrust::counting_iterator<unsigned int> end(vec.size() );
	thrust::tuple<float, unsigned int> init(vec[0], 0);
	thrust::tuple<float, unsigned int> smallest;

	smallest = reduce( thrust::make_zip_iterator(make_tuple(vec.begin(), begin) ),
			thrust::make_zip_iterator(make_tuple(vec.end(), end) ),
			init,
			bigger_tuple_functor() );

	ID = thrust::get<1>(smallest);
	return vec[ID];
}

float hostGetMin(const thrust::device_vector<float>& vec, unsigned int &ID) {
	// create implicit index sequence [0, 1, 2, ... ]
	thrust::counting_iterator<unsigned int> begin(0);
	thrust::counting_iterator<unsigned int> end(vec.size() );
	thrust::tuple<float, unsigned int> init(vec[0], 0);
	thrust::tuple<float, unsigned int> smallest;
	
	smallest = reduce( thrust::make_zip_iterator(make_tuple(vec.begin(), begin) ),
			thrust::make_zip_iterator(make_tuple(vec.end(), end) ),
			init,
			smaller_tuple_functor() );

	ID = thrust::get<1>(smallest);
	return vec[ID];
}
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Layout of SOMEdgeF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		toNeur1	toNeur1	toNeur1	..
 * ROW2		toNeur2	toNeur2	toNeur2	..
 * ROW3		toNeur3	toNeur3	toNeur3	..
 * ROW(n+1)	..		..		..
 */
BMUExport
hostSOMFindBMNeuronID(std::vector<SplittedNetExport*> &SExp,
		const float &fConscienceRate)
{
	BMUExport retBMU;
	float fLastBMU = FLT_MAX;

	omp_set_num_threads(SExp.size() );  	// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 			//for(int iDev = 0; iDev < static_cast<int>(SExp.size() ); iDev++) {
	{
		unsigned int iDev = omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDev) );
		unsigned int BMUID = 0;

		unsigned int iWidth 	= SExp.at(iDev)->f2dEdges.GetW();
		unsigned int iHeight 	= SExp.at(iDev)->f2dEdges.GetH();

		assert(iWidth 	> 0);
		assert(iHeight 	> 0);

		thrust::device_vector<float> dvRes(iWidth, 0.f);
		thrust::device_vector<float> dvTmp(iWidth, 0.f); 			// temporary
		thrust::device_vector<float> dvConscience(iWidth, 1.f / (float)iWidth);

		for(unsigned int y = 0; y < iHeight; y++) {
			thrust::transform(
				SExp.at(iDev)->f2dEdges.GetRowBegin(y),			// input
				SExp.at(iDev)->f2dEdges.GetRowEnd(y), 			// input
				dvTmp.begin(), 						// result
				minus_pow_functor((*SExp.at(iDev)->dvInput)[y]) ); 	// functor

			thrust::transform(
				dvRes.begin(), 						// input
				dvRes.end(), 						// input
				dvTmp.begin(),						// input
				dvRes.begin(), 						// result
				thrust::plus<float>() );				// functor
		}

		// implementation of conscience mechanism
		if(fConscienceRate > 0.f) {
			thrust::transform(
				dvConscience.begin(),
				dvConscience.end(),
				SExp.at(iDev)->dvConscience->begin(),
				dvConscience.begin(),
				thrust::minus<float>() );

			thrust::transform(
				dvRes.begin(),
				dvRes.end(),
				dvConscience.begin(),
				dvRes.begin(),
				thrust::minus<float>() );
		}

		thrust::transform(
			dvRes.begin(),
			dvRes.end(),
			SExp.at(iDev)->dvConscience->begin(),
			SExp.at(iDev)->dvConscience->begin(),
			saxmy_functor(fConscienceRate) );

		hostGetMin(dvRes, BMUID);

		// Check partial results for global BMU in all devices
		if(fLastBMU > dvRes[BMUID]) {
			fLastBMU = dvRes[BMUID];
			thrust::host_vector<float> vPos = SExp.at(iDev)->f2dPositions.GetSubArrayY(BMUID);
			retBMU = BMUExport(BMUID, iDev, vPos);
		}
	}
	return retBMU;
}

/*
 * Layout of SOMPositionF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		Xpos	Xpos	Xpos	..
 * ROW2		Ypos	Ypos	Ypos	..
 * ROW3		Zpos	Zpos	Zpos	..
 * ROW(n+1)	..		..		..		..
 */
template<typename BinaryFunction>
void hostSOMPropagateBW( std::vector<SplittedNetExport*> &SExp,
		const BMUExport &BMU,
		const float &fSigmaT,
		const float &fLearningRate,
		const BinaryFunction &binaryDistFunc
		)
{
	omp_set_num_threads(SExp.size() );  	// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 			//for(int iDev = 0; iDev < static_cast<int>(SExp.size() ); iDev++) {
	{
		unsigned int iDev = omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDev) );
		
		unsigned int iWidth 	= SExp.at(iDev)->f2dPositions.GetW();
		unsigned int iHeight 	= SExp.at(iDev)->f2dPositions.GetH();

		thrust::device_vector<float> dvTmp(iWidth, 0.f); 			// temporary
		thrust::device_vector<float> dvInfluence(iWidth, 0.f);
		thrust::device_vector<float> dvDist(iWidth, 0.f);

		// 1. Calc distances for all neurons to BMNeuron
		// Distance = sqrt(pow(x,2)+pow(y,2)+pow(z,2)+pow(n+1,2) );
		for(int y = 0; y < static_cast<int>(iHeight); y++) { 				// for each coordinate position of the neuron
			thrust::transform(
				SExp.at(iDev)->f2dPositions.GetRowBegin(y),		// input
				SExp.at(iDev)->f2dPositions.GetRowEnd(y), 		// input
				dvTmp.begin(), 						// result
				minus_pow_functor(BMU.dvBMUPos[y]) ); 			// functor

			thrust::transform(
				dvDist.begin(), 					// input
				dvDist.end(), 						// input
				dvTmp.begin(),						// input
				dvDist.begin(), 					// result
				thrust::plus<float>() );				// functor
		}

		thrust::transform(
			dvDist.begin(),							// input
			dvDist.end(), 							// input
			dvDist.begin(), 						// result
			sqrt_functor() );						// functor

		// 2. Calculate the influence for each neuron
		thrust::transform(
			dvDist.begin(),							// input
			dvDist.end(), 							// input
			dvInfluence.begin(), 						// result
			binaryDistFunc );						// functor

		// 3. Only handle neurons in radius:
		// 3a. Make stencil
		dvTmp.assign(iWidth, fSigmaT);
		thrust::transform(
			dvDist.begin(), 						// input 1
			dvDist.end(),							// input 1
			dvTmp.begin(),							// input 1
			dvTmp.begin(), 							// result
			thrust::less<float>() 						// functor
		);

		// 3b. Use stencil to modify only neurons inside the radius
		// Save result in the ANN::F2DArray
		iWidth 	= SExp.at(iDev)->f2dEdges.GetW();
		iHeight = SExp.at(iDev)->f2dEdges.GetH();

		for(int y = 0; y < static_cast<int>(iHeight); y++) {				// for each edge of the neuron
			thrust::transform_if(
				SExp.at(iDev)->f2dEdges.GetRowBegin(y),			// input 1
				SExp.at(iDev)->f2dEdges.GetRowEnd(y), 			// input 1
				dvInfluence.begin(),					// input 2
				dvTmp.begin(),						// stencil
				SExp.at(iDev)->f2dEdges.GetRowBegin(y), 		// result
				hebbian_functor(fLearningRate, (*SExp.at(iDev)->dvInput)[y]), // functor
				thrust::identity<int>() ); 				// predicate
		}
	}
}

void hostSOMTraining( std::vector<SplittedNetExport*> &SExp,
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles,
		const float &fSigma0, 
		const float &fLearningRate0,
		const float &fConscienceRate,
		float (*pfnDecay)(const float &, const float &, const float &),
		const ANN::DistFunction &DistFunc )
{
	float fLambda 	= iCycles / log(fSigma0);

	int iMin 		= 0;
	int iMax 		= InputSet.GetNrElements()-1;
	unsigned int iProgCount = 1;

	// use 8 proximal neurons as standard
	float fSigmaT = sqrt(2.f);

	for(unsigned int i = 0; i < iCycles; i++) {
		if(iCycles >= 10) {
			if(((i+1) / (iCycles/10)) == iProgCount && (i+1) % (iCycles/10) == 0) {
				std::cout<<"Current training progress calculated by the GPU is: "<<iProgCount*10.f<<"%/Step="<<i+1<<std::endl;
				iProgCount++;
			}
		}
		else {
			std::cout<<"Current training progress calculated by the CPU is: "<<(float)(i+1.f)/(float)iCycles*100.f<<"%/Step="<<i+1<<std::endl;
		}

		// Set input
		std::vector<float> vCurInput = InputSet.GetInput(ANN::RandInt(iMin, iMax) );
		
		for(int iDev = 0; iDev < static_cast<int>(SExp.size() ); iDev++) {
			checkCudaErrors(cudaSetDevice(iDev) );
			thrust::device_vector<float> *p_dvInputVector = new thrust::device_vector<float>(vCurInput.size() );
			thrust::copy(vCurInput.begin(), vCurInput.end(), p_dvInputVector->begin() );
			SExp[iDev]->dvInput = p_dvInputVector;
		}

		// Find BMNeuron
		BMUExport BMUExp = hostSOMFindBMNeuronID(SExp, fConscienceRate);

		// Calc m_fSigmaT if conscience is _not_ used
		if(fConscienceRate <= 0.f) {
			fSigmaT = std::floor(pfnDecay(fSigma0, i, fLambda) + 0.5f);
		}
		float fLearningRate = pfnDecay(fLearningRate0, i, iCycles);

		// Propagate BW
		if (strcmp (DistFunc.name, "gaussian") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,				// const
					fSigmaT,			// const
					fLearningRate,
					gaussian_functor(fSigmaT)); 	// const
		}
		else if (strcmp (DistFunc.name, "mexican") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,				// const
					fSigmaT,			// const
					fLearningRate,
					mexican_functor(fSigmaT)); 	// const
		}
		else if (strcmp (DistFunc.name, "bubble") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,				// const
					fSigmaT,			// const
					fLearningRate,
					bubble_functor(fSigmaT)); 	// const
		}
		else if (strcmp (DistFunc.name, "cut_gaussian") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,				// const
					fSigmaT,			// const
					fLearningRate,
					cut_gaussian_functor(fSigmaT)); // const
		}
		else if (strcmp (DistFunc.name, "epanechicov") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,				// const
					fSigmaT,			// const
					fLearningRate,
					epanechicov_functor(fSigmaT)); 	// const
		}
	}
}

#endif
