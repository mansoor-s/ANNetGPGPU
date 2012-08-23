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

// Y <- A * X * Y
struct sax_functor {
    const float a;

    sax_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(const float& x) const {
		return a * x;
	}
};

// NEEDED: -arch=sm_20
struct logTransferFcn {
    __host__ __device__
	float operator()(const float& fVal, const float& fBias) const { 
    	return ANN::fcn_log_normal(fVal, fBias);
	}
};

struct devLogTransferFcn {
    __host__ __device__
	float operator()(const float& fVal) const { 
    	return ANN::fcn_log_derivate(fVal, 0);
	}
};

/*
 * THIS FUNCTION WORKS
 */
std::vector<float>
hostBPCalcDelta(	const thrust::device_vector<float> &dvNeurOut,	// from forward run
					const std::vector<float> &vTrainOut ) 			// from training set
{
	thrust::device_vector<float> dvTrainOut (vTrainOut.begin(), vTrainOut.end() );
	thrust::device_vector<float> dvDelta	(vTrainOut.size(), 0.f);
    std::vector<float> vRes(vTrainOut.size() );
    
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

/*
 * THIS FUNCTION WORKS AS WELL
 */
std::vector<thrust::device_vector<float> >
hostBPPropagateFW(	const std::vector<ANN::Matrix> &vEdgeMatrices,
					const std::vector<ANN::Matrix> &vBiasEdgeMatrices,
					const std::vector<float> &vInput)
{
	std::vector<thrust::device_vector<float> > vNeuronValues(1, vInput);

	// Copy Input from vInput in device vector: vOutput
	thrust::host_vector<float> hvInput(vInput.begin(), vInput.end() ); 	// input
	thrust::device_vector<float> dvLayer;
	thrust::device_vector<float> dvBias;
	
	unsigned int iWidth 	= 0;
	unsigned int iHeight 	= 0;

	for(unsigned int i = 0; i < vEdgeMatrices.size(); i++) {	
		iWidth 		= vEdgeMatrices.at(i).getW();							// iWidth == size of the next layer
		iHeight 	= vEdgeMatrices.at(i).getH();							// iHeight == size of this layer
		
		// Alloc memory
		dvLayer 	= thrust::device_vector<float>(iWidth, 0.f); 			// iWidth == size of the next layer
	
		// Calculate the result of the current layer
		for(unsigned int y = 0; y < iHeight; y++) {	
		    // Y <- A * X + Y
		    thrust::transform(
		    		vEdgeMatrices.at(i).getRowBegin(y),
		    		vEdgeMatrices.at(i).getRowEnd(y),
		    		dvLayer.begin(),
		    		dvLayer.begin(),
		    		saxpy_functor(hvInput[y]) );
		}
		
		// TODO optimize that
		if(vBiasEdgeMatrices.size() > 0) {
			dvBias = thrust::device_vector<float>(vBiasEdgeMatrices.at(i).getRowBegin(0), vBiasEdgeMatrices.at(i).getRowEnd(0));
		}
		else {
			dvBias 	= thrust::device_vector<float>(iWidth, 0.f); 			// bias
		}

		// Run values through transfer function
	    thrust::transform(
	    		dvLayer.begin(),
	    		dvLayer.end(),
	    		dvBias.begin(),
	    		dvLayer.begin(),
	    		logTransferFcn() );

		// Now the input of the next layer will be the the previous one
		hvInput = dvLayer;
		vNeuronValues.push_back(dvLayer);
	}
	return vNeuronValues;
}

std::vector<ANN::Matrix>
hostBPPropagateBW(	const std::vector<ANN::Matrix> &vEdgeMatricesO,
					std::vector<ANN::Matrix> &vEdgeMatricesI,
					std::vector<thrust::device_vector<float> > vErrors,
					const std::vector<thrust::device_vector<float> > &vNeuronValues,
					const float &fLearningRate)
{
	// Create matrix for momentums
	std::vector<ANN::Matrix> vEdgeMomentums;
	for(unsigned int i = 0; i < vEdgeMatricesI.size(); i++) {
		ANN::Matrix matrix(vEdgeMatricesI.at(i).getW(), vEdgeMatricesI.at(i).getH(), 0.f);
		vEdgeMomentums.push_back(matrix);
	}
	
	for(int i = vEdgeMatricesO.size()-1; i >= 0; i--) {						// All layers except output!
		unsigned int iWidth 	= vEdgeMatricesO.at(i).getW();							// Nr. of neurons in next layer
		unsigned int iHeight 	= vEdgeMatricesO.at(i).getH();							// Nr. of neurons in this layer
		
		if(vErrors.front().size() == iHeight) {
			// Calculate the result of the current layer
			for(unsigned int y = 0; y < iHeight; y++) {	
				thrust::transform(
				vEdgeMatricesO.at(i).getRowBegin(y),
				vEdgeMatricesO.at(i).getRowEnd(y),
				vErrors.at(i).begin(),
				vErrors.at(i).begin(),
				saxpy_functor(vErrors.at(i+1)[y]) ); // TODO
			}

			// Run values through transfer function
			thrust::transform(
					vErrors.at(i).begin(),
					vErrors.at(i).end(),
					vErrors.at(i).begin(),
					devLogTransferFcn() );
		}
	}
	
	// All layers except output ..
	for(int i = vEdgeMatricesI.size()-1; i >= 0 && vNeuronValues.size() > 0; i--) {
		unsigned int iWidth 	= vEdgeMatricesI.at(i).getW();							// Nr. of neurons in next layer
		unsigned int iHeight 	= vEdgeMatricesI.at(i).getH();							// Nr. of neurons in this layer

		for(unsigned int x = 0; x < iHeight; x++) {
			thrust::device_vector<float> dvMomentums(iWidth, 0.f);

			// standard back propagation algorithm
				// fVal = pCurEdge->GetDestination(this)->GetErrorDelta() * m_fLearningRate * GetValue()
			// TODO weight decay term
				// - m_fWeightDecay * pCurEdge->GetValue()
			// TODO momentum term
				// + m_fMomentum * pCurEdge->GetMomentum();

			thrust::transform(
					vErrors.at(i+1).begin(),//vErrorDeltas.at(i+1).begin(),
					vErrors.at(i+1).end(),//vErrorDeltas.at(i+1).end(),
					dvMomentums.begin(),
					sax_functor(fLearningRate*vNeuronValues.at(i)[x]) );

			thrust::transform(
					dvMomentums.begin(),
					dvMomentums.end(),
					vEdgeMatricesI.at(i).getRowBegin(x),
					vEdgeMatricesI.at(i).getRowBegin(x),
					thrust::plus<float>() );
		}
	}
	
	return vEdgeMomentums;
}

#endif
