#ifndef _BPKERNELS_
#define _BPKERNELS_

#include <math/ANFunctions.h>
#include <gpgpu/ANKernels.h>


// Y <- A * X + Y
struct saxpy_functor {
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
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
hostBPPropagateFW(	const std::vector<ANN::Matrix> &vEdgeMatrices,
					const std::vector<ANN::Matrix> &vBiasEdgeMatrices,
					const std::vector<float> &vInput, 
					float (*pf_Transfer)(const float &, const float &) ) 
{
	// Output vector
	std::vector<float> vOutput;
	
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
	}
	
	// Copy device layer dvLayer in vOutput
	thrust::copy(dvLayer.begin(), dvLayer.end(), vOutput.begin()); 			// output
	return vOutput;
}

void 
hostBPPropagateBW(	const std::vector<ANN::Matrix> &vEdgeMatrices,
					const std::vector<float> &vOutput, 						// from hostBPPropagateFW()
					const std::vector<float> &vTrainOut,					// from training set
					float (*pf_DevTransfer)(const float &, const float &) ) // deviation of transfer function
{
	thrust::device_vector<float> dvOutput	(vOutput.begin(), vOutput.end() ); 
	thrust::device_vector<float> dvTrainOutp(vTrainOut.begin(), vTrainOut.end() );
	thrust::device_vector<float> dvErrors	(vOutput.size(), 0.f);
	
	std::vector<thrust::device_vector<float> > vErrorDeltas;
	
//	unsigned int iWidth 	= 0;
	unsigned int iHeight 	= 0;
	
	// Calc error deltas of output layer
    thrust::transform(
    		dvTrainOutp.begin(),
    		dvTrainOutp.end(),
    		dvOutput.begin(),
    		dvErrors.begin(),
    		thrust::minus<float>() ); 
    // save in vector
    vErrorDeltas.push_back(dvErrors);
    
	// calc it for the rest of the layers
	for(int i = vEdgeMatrices.size()-1; i >= 0; i--) {					// All layers except output!
//		iWidth 	= vEdgeMatrices.at(i).GetW();							// Nr. of neurons in next layer
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
	    vErrorDeltas.push_back(dvErrors);
	}
	
	// Calc new weight matrix
	for(int i = vEdgeMatrices.size()-1; i >= 0; i--) {					// All layers except output!
		// TODO
	}
}

#endif
