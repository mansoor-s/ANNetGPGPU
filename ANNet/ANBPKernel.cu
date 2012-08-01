#ifndef _BPKERNELS_
#define _BPKERNELS_

#include <math/ANFunctions.h>
#include <gpgpu/ANKernels.h>

// X <= Y
struct equaly_functor {
    __host__ __device__
        float operator()(const float& y) const { 
            return y;
        }
};

// Y <- A * X + Y
struct saxpy_functor {
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

// Y <- A * X * Y
struct saxy_functor {
    const float a;

    saxy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x * y;
        }
};

// NEEDED: -arch=sm_20
struct binaryTransferFcn_functor {
	float (*m_pf)(const float &, const float &);
	
	binaryTransferFcn_functor(float (*pf)(const float &, const float &) ) : m_pf(pf)	{}
	
    __host__ __device__
	float operator()(const float& fVal, const float& fBias) const { 
		return (m_pf)(fVal, fBias);
	}
};

struct unaryTransferFcn_functor {
	float (*m_pf)(const float &, const float &);
	
	unaryTransferFcn_functor(float (*pf)(const float &, const float &) ) : m_pf(pf)	{}
	
    __host__ __device__
	float operator()(const float& fVal) const { 
		return (m_pf)(fVal, 0);
	}
};

std::vector<float>
hostBPCalcDelta(	const thrust::device_vector<float> &dvNeurOut,	// from forward run
					const std::vector<float> &vTrainOut ) 			// from training set
{
	thrust::device_vector<float> dvTrainOut (vTrainOut.begin(), vTrainOut.end() );
	thrust::device_vector<float> dvDelta	(dvNeurOut.size(), 0.f);
    std::vector<float> vRes;
	
	// Calc error deltas of output layer
    thrust::transform(
    		dvTrainOut.begin(),
    		dvTrainOut.end(),
    		dvNeurOut.begin(),
    		dvDelta.begin(),
    		thrust::minus<float>() ); 

    thrust::copy(dvDelta.begin(), dvDelta.end(), vRes.begin());
    return vRes;
}

std::vector<thrust::device_vector<float> >
hostBPPropagateFW(	const std::vector<ANN::Matrix> &vEdgeMatrices,
					const std::vector<ANN::Matrix> &vBiasEdgeMatrices,
					const std::vector<float> &vInput, 
					float (*pf_Transfer)(const float &, const float &) ) 
{
	std::vector<thrust::device_vector<float> > vNeuronValues;
	
	// Copy Input from vInput in device vector: vOutput
	thrust::device_vector<float> dvInput(vInput.begin(), vInput.end() ); 	// input
	thrust::device_vector<float> dvLayer;
	
	unsigned int iWidth 	= 0;
	unsigned int iHeight 	= 0;
	
	for(unsigned int i = 0; i < vEdgeMatrices.size(); i++) {
		iWidth 		= vEdgeMatrices.at(i).GetW();							// Nr. of neurons in next layer
		iHeight 	= vEdgeMatrices.at(i).GetH();							// Nr. of neurons in this layer
		
		// alloc memory
		dvLayer 	= thrust::device_vector<float>(iWidth, 0.f); 			// layer
		
		// Calculate the result of the current layer
		for(unsigned int y = 0; y < iHeight; y++) {	
		    // Y <- A * X + Y
		    thrust::transform(
		    		vEdgeMatrices.at(i).getRowBegin(y), 					// X
		    		vEdgeMatrices.at(i).getRowEnd(y), 						// X
		    		dvLayer.begin(), 										// Y
		    		dvLayer.begin(), 										// Y
		    		saxpy_functor(dvInput[y]) ); 							// A
			
		}
		
		// Run values through transfer function
	    thrust::transform(
	    		dvLayer.begin(),
	    		dvLayer.end(),
	    		vBiasEdgeMatrices.at(i).getRowBegin(0),
	    		dvLayer.begin(),
	    		binaryTransferFcn_functor(pf_Transfer) ); 
		
		// Now the input of the next layer will be the the previous one
		dvInput = dvLayer;
		vNeuronValues.push_back(dvLayer);
	}
	
	return vNeuronValues;
}

std::vector<ANN::Matrix>
hostBPPropagateBW(	std::vector<ANN::Matrix> &vEdgeMatrices,
					const std::vector<thrust::device_vector<float> > &vNeuronValues,
					const std::vector<float> &vErrors, 						// 
					const float &fLearningRate, 							// learning rate
					float (*pf_DevTransfer)(const float &, const float &) ) // deviation of transfer function
{
	// Create matrix for momentums
	std::vector<ANN::Matrix> vEdgeMomentums;
	for(unsigned int i = 0; i < vEdgeMatrices.size(); i++) {
		ANN::Matrix matrix(vEdgeMatrices.at(i).GetW(), vEdgeMatrices.at(i).GetH(), 0.f);
		vEdgeMomentums.push_back(matrix);
	}
	
	thrust::device_vector<float> dvErrors	(vErrors.begin(), vErrors.end());
	std::vector<thrust::device_vector<float> > vErrorDeltas;
    vErrorDeltas.push_back(dvErrors);
	
	unsigned int iHeight 	= 0;
	// calc it for the rest of the layers
	for(int i = vEdgeMatrices.size()-1; i >= 0; i--) {					// All layers except output!
		iHeight = vEdgeMatrices.at(i).GetH();							// Nr. of neurons in this layer
		
		dvErrors = thrust::device_vector<float>(iHeight, 0.f);
		
		// Calculate the result of the current layer
		for(unsigned int y = 0; y < iHeight; y++) {	
			// Y <- A * X + Y
		    thrust::transform(
		    		vEdgeMatrices.at(i).getRowBegin(y), 					// X
		    		vEdgeMatrices.at(i).getRowEnd(y), 						// X
		    		dvErrors.begin(), 										// Y
		    		dvErrors.begin(), 										// Y
		    		saxpy_functor(vErrorDeltas.back()[y]) ); 							// A
		}
		
		// Run values through transfer function
	    thrust::transform(
	    		dvErrors.begin(),
	    		dvErrors.end(),
	    		dvErrors.begin(),
	    		unaryTransferFcn_functor(pf_DevTransfer) ); 
	    // save in vector
	    vErrorDeltas.insert(vErrorDeltas.begin(), dvErrors);
	}
	
	// Calc new weight matrix
	// Adapt weights
	for(int i = vEdgeMatrices.size()-1; i >= 0; i--) {					// All layers except output!
		iHeight = vEdgeMatrices.at(i).GetH();							// Nr. of neurons in this layer
		
		thrust::device_vector<float> dvMomentums(vNeuronValues.at(i).size(), 0.f);
		
		// M <- A * X * Y
		thrust::transform(
				vErrorDeltas.at(i+1).begin(), 							// X
				vErrorDeltas.at(i+1).end(), 							// X
				vNeuronValues.at(i).begin(), 							// Y
				dvMomentums.begin(), 									// Y
				saxy_functor(fLearningRate) ); 							// A
		
		// Calculate the result of the current layer
		for(unsigned int y = 0; y < iHeight; y++) {		
			// X <- Y
			thrust::transform(
					dvMomentums.begin(), 								// Y
					dvMomentums.end(), 									// Y
					vEdgeMomentums.at(i).getRowBegin(y), 				// X
					equaly_functor() ); 								// X <- Y
			
			// Y <- A + B
			thrust::transform(
					dvMomentums.begin(), 								// Y
					dvMomentums.end(), 									// Y
					vEdgeMatrices.at(i).getRowBegin(y), 				// X
					vEdgeMatrices.at(i).getRowBegin(y), 				// X
					thrust::plus<float>() );
		}
	}
	
	return vEdgeMomentums;
}

#endif
