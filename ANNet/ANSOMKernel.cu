#ifndef _SOMKERNELS_
#define _SOMKERNELS_

#include <math/ANFunctions.h>
#include <math/ANRandom.h>
#include <gpgpu/ANKernels.h>

#include <cassert>
#include <cmath>


struct saxmy_functor {
	const float a;

	saxmy_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(const float& x, const float& y) const { 
		return a * (x - y);
	}
};


// return the biggest of two tuples
struct bigger_tuple_functor {
    __device__ __host__
    thrust::tuple<float, unsigned int> operator() (	
    	const thrust::tuple<float, unsigned int> &a, 
		const thrust::tuple<float, unsigned int> &b ) 
    {
    	return (a >= b) ? a : b;
    }
};

// return the biggest of two tuples
struct smaller_tuple_functor {
    __device__ __host__
    thrust::tuple<float, unsigned int> operator() (	
    	const thrust::tuple<float, unsigned int> &a, 
		const thrust::tuple<float, unsigned int> &b ) 
    {
    	return (a <= b) ? a : b;
    }
};

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

/**
 * 
 */
struct minus_pow_functor {
    const float fVal;
    minus_pow_functor(float val) : fVal(val) {}

    __host__ __device__
	float operator()(const float& val) const { 
		return pow(fVal-val, 2);
	}
};

struct sqrt_functor {
    __host__ __device__
	float operator()(const float& val) const { 
		return sqrt(val);
	}
};

/*
 * Layout of SOMEdgeMatrix:
 * 			COL1	COL2	COL3	COL(n+1)
 * ROW1		toNeur1	toNeur1	toNeur1	..
 * ROW2		toNeur2	toNeur2	toNeur2	..
 * ROW3		toNeur3	toNeur3	toNeur3	..
 * ROW(n+1)	..		..		..
 */
unsigned int hostSOMFindBMNeuronID( thrust::device_vector<float> &ConscienceVector,
		const ANN::Matrix &SOMEdgeMatrix, 
		const thrust::device_vector<float> &InputVector,
		const float &fConscienceRate) 
{
	unsigned int BMUID 		= 0;
	unsigned int iWidth 	= SOMEdgeMatrix.getW();
	unsigned int iHeight 	= SOMEdgeMatrix.getH();
	
	assert(iWidth > 0);
	assert(iHeight > 0);
	
	thrust::device_vector<float> dvRes(iWidth, 0.f);
	thrust::device_vector<float> dvConscience(iWidth, -1.f / (float)iWidth);
	thrust::device_vector<float> dvTmp(iWidth, 0.f); // temporary
	
	for(unsigned int y = 0; y < iHeight; y++) {
		thrust::transform(
				SOMEdgeMatrix.getRowBegin(y),		// input
				SOMEdgeMatrix.getRowEnd(y), 		// input
				dvTmp.begin(), 						// result
				minus_pow_functor(InputVector[y]) ); // functor

		thrust::transform(
				dvRes.begin(), 						// input
				dvRes.end(), 						// input
				dvTmp.begin(),						// input
				dvRes.begin(), 						// result
				thrust::plus<float>() );			// functor
	}

	// implementation of conscience mechanism
	dvTmp = dvRes;
	if(fConscienceRate > 0.f) {
		thrust::transform(
			ConscienceVector.begin(), 
			ConscienceVector.end(), 
			dvConscience.begin(), 
			dvConscience.begin(), 
			thrust::plus<float>() );

		thrust::transform(
				dvConscience.begin(),
				dvConscience.end(),
				dvRes.begin(),
				dvRes.begin(),
				thrust::plus<float>() );
	}

	thrust::transform(
		dvTmp.begin(),
		dvTmp.end(),
		ConscienceVector.begin(),
		ConscienceVector.begin(),
		saxmy_functor(fConscienceRate) );

/*
	thrust::transform(
			dvRes.begin(),							// input
			dvRes.end(), 							// input
			dvRes.begin(), 							// result
			sqrt_functor() );						// functor
*/
	hostGetMin(dvRes, BMUID);
	
//	dvRes.clear();									// cleanup
//	dvTmp.clear();									// cleanup
	
	return BMUID;
}

/* // TODO fucking need better GRAKA
struct dist_functor {
	float fSigmaT;
	float (*pf_distance)(const float &, const float &);
	dist_functor(float (*distance)(const float &, const float &), const float &sigmaT) : pf_distance(distance), fSigmaT(sigmaT)	{}
	
    __host__ __device__
	float operator()(const float& val) const { 
		return (pf_distance)(val, fSigmaT);
	}
};
*/ // TODO fucking need better GRAKA

struct gaussian_bell_functor {
	float fSigmaT;
	gaussian_bell_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}
	
    __host__ __device__
	float operator()(const float& dist) const { 
    	return ANN::fcn_gaussian_bell(dist, fSigmaT);
	}
};

struct hebbian_functor {
	float fLearningRate;
	float fInput;
	
	hebbian_functor(const float &learning_rate, const float &input) : 
		fLearningRate(learning_rate), fInput(input) {}
	
    __host__ __device__
	float operator()(const float& fWeight, const float& fInfluence) const { 
    	return fWeight + (fInfluence*fLearningRate*(fInput-fWeight) );
	}
};

/*
 * Layout of SOMPositionMatrix:
 * 			COL1	COL2	COL3	COL(n+1)
 * ROW1		Xpos	Xpos	Xpos	..
 * ROW2		Ypos	Ypos	Ypos	..
 * ROW3		Zpos	Zpos	Zpos	..
 * ROW(n+1)	..		..		..		..
 */
void hostSOMPropagateBW( ANN::Matrix &SOMEdgeMatrix,
		const ANN::Matrix &SOMPositionMatrix, 
		const thrust::device_vector<float> &dvInputVector,
		const unsigned int BMUID, 
		const float &fSigmaT, 
		const float &fLearningRate
		) 
{
	unsigned int iWidth 	= SOMPositionMatrix.getW();
	unsigned int iHeight 	= SOMPositionMatrix.getH();
	
	// TODO PUT this in the TRAINING function the increase performance
	thrust::device_vector<float> dvBMUPos = SOMPositionMatrix.getCol(BMUID);
	thrust::device_vector<float> dvTmp(iWidth, 0.f); // temporary
	thrust::device_vector<float> dvInfluence(iWidth, 0.f); 
	thrust::device_vector<float> dvDist(iWidth, 0.f);
	
	// 1. Calc distances for all neurons to BMNeuron
	// Distance = sqrt(pow(x,2)+pow(y,2)+pow(z,2)+pow(n+1,2) );
	for(unsigned int y = 0; y < iHeight; y++) { 	// for each coordinate position of the neuron
		thrust::transform(
				SOMPositionMatrix.getRowBegin(y),	// input
				SOMPositionMatrix.getRowEnd(y), 	// input
				dvTmp.begin(), 						// result
				minus_pow_functor(dvBMUPos[y]) ); 	// functor
		
		thrust::transform(
				dvDist.begin(), 					// input
				dvDist.end(), 						// input
				dvTmp.begin(),						// input
				dvDist.begin(), 					// result
				thrust::plus<float>() );			// functor
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
			dvInfluence.begin(), 					// result
			gaussian_bell_functor(fSigmaT) );		// functor
	
	// 3. Only handle neurons in radius:
	// 3a. Make stencil
	dvTmp.assign(iWidth, fSigmaT);
	thrust::transform(
			dvDist.begin(), 						// input 1
			dvDist.end(),							// input 1
			dvTmp.begin(),							// input 1
			dvTmp.begin(), 							// result
			thrust::less_equal<float>() 			// functor
	);
	
	// 3b. Use stencil to modify only neurons inside the radius
	// Save result in the ANN::Matrix
	iWidth 	= SOMEdgeMatrix.getW();
	iHeight = SOMEdgeMatrix.getH();

	for(unsigned int y = 0; y < iHeight; y++) {		// for each edge of the neuron   	
		thrust::transform_if(
				SOMEdgeMatrix.getRowBegin(y),		// input 1
				SOMEdgeMatrix.getRowEnd(y), 		// input 1
				dvInfluence.begin(),				// input 2
				dvTmp.begin(),						// stencil
				SOMEdgeMatrix.getRowBegin(y), 		// result
				hebbian_functor(fLearningRate, dvInputVector[y]), // functor
				thrust::identity<int>() ); 			// predicate
	}
	
	// 4. Clean!
//	dvBMUPos.clear();
//	dvTmp.clear(); 									// cleanup
//	dvInfluence.clear(); 							// cleanup
//	dvDist.clear(); 								// cleanup
}

void hostSOMTraining( thrust::device_vector<float> &ConscienceVector,
		ANN::Matrix &SOMEdgeMatrix,
		const ANN::Matrix &SOMPositionMatrix, 
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles,
		const float &fSigma0, 
		const float &fLearningRate0,
		const float &fConscienceRate,
		float (*pfnDecay)(const float &, const float &, const float &) )
{
	float fLambda 	= iCycles / log(fSigma0);
	
	int iMin 		= 0;
	int iMax 		= InputSet.GetNrElements()-1;
	unsigned int iProgCount = 1;
	
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
		thrust::device_vector<float> dvInputVector(vCurInput.size() );
		thrust::copy(vCurInput.begin(), vCurInput.end(), dvInputVector.begin() );
		
		// Find BMNeuron
		unsigned int BMUID = hostSOMFindBMNeuronID(ConscienceVector, SOMEdgeMatrix, dvInputVector, fConscienceRate);

		// use 8 proximal neurons as standard 
		float fSigmaT = sqrt(2.f);
		// Calc m_fSigmaT if conscience is _not_ used
		if(fConscienceRate == 0.f)
			fSigmaT = pfnDecay(fSigma0, i, fLambda);
		float fLearningRate = pfnDecay(fLearningRate0, i, iCycles);
		
		// Propagate BW
		hostSOMPropagateBW( SOMEdgeMatrix,
				SOMPositionMatrix, 	// const
				dvInputVector,		// const
				BMUID,			// const
				fSigmaT,		// const
				fLearningRate ); 	// const
	}
}

#endif
