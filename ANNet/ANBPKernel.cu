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
struct binaryTransferFcn_functor {
	binaryTransferFcn_functor(const ANN::TransfFunction &function) {
	}
	
    __host__ __device__
	float operator()(const float& fVal, const float& fBias) const { 
    	return ANN::fcn_log_normal(fVal, fBias);
	}
};

struct unaryTransferFcn_functor {
	unaryTransferFcn_functor(const ANN::TransfFunction &function) {
	}
	
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

    for(int i = 0; i < dvNeurOut.size(); i++) {
    	std::cout<<"NEURON at "<<i<<": "<<dvNeurOut[i]<<std::endl;
    }
    for(int i = 0; i < vRes.size(); i++) {
    	std::cout<<"DELTA at "<<i<<": "<<vRes[i]<<std::endl;
    }

    return vRes;
}

/*
 * THIS FUNCTION WORKS AS WELL
 */
std::vector<thrust::device_vector<float> >
hostBPPropagateFW(	const std::vector<ANN::Matrix> &vEdgeMatrices,
					const std::vector<ANN::Matrix> &vBiasEdgeMatrices,
					const std::vector<float> &vInput, 
					const ANN::TransfFunction &function)
{
	std::vector<thrust::device_vector<float> > vNeuronValues(1, vInput);

	// Copy Input from vInput in device vector: vOutput
	thrust::host_vector<float> hvInput(vInput.begin(), vInput.end() ); 	// input
	thrust::device_vector<float> dvLayer;
	thrust::device_vector<float> dvBias;
	
	unsigned int iWidth 	= 0;
	unsigned int iHeight 	= 0;

	/*
	for(int i = 0; i < hvInput.size(); i++) {
		std::cout<<"i: "<<hvInput[i]<<std::endl;
	}
	for(unsigned int i = 0; i < vEdgeMatrices.size(); i++) {
		for(unsigned int j = 0; j < vEdgeMatrices.at(i).GetW(); j++) {
			for(unsigned int k = 0; k < vEdgeMatrices.at(i).GetH(); k++) {
				std::cout<<"MAT: "<<vEdgeMatrices.at(i)[j*vEdgeMatrices.at(i).GetW()+k]<<std::endl;
			}
		}
	}
	*/

	for(unsigned int i = 0; i < vEdgeMatrices.size(); i++) {	
		iWidth 		= vEdgeMatrices.at(i).getW();							// iWidth == size of the next layer
		iHeight 	= vEdgeMatrices.at(i).getH();							// iHeight == size of this layer
		
		// alloc memory
		dvLayer 	= thrust::device_vector<float>(iWidth, 0.f); 			// iWidth == size of the next layer
	
		// Calculate the result of the current layer
		for(unsigned int y = 0; y < iHeight; y++) {	
		    // Y <- A * X + Y
		    thrust::transform(
		    		vEdgeMatrices.at(i).getRowBegin(y), 					// X
		    		vEdgeMatrices.at(i).getRowEnd(y), 						// X
		    		dvLayer.begin(), 										// Y
		    		dvLayer.begin(), 										// Y
		    		saxpy_functor(hvInput[y]) ); 							// A
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
	    		binaryTransferFcn_functor(function) );

		// Now the input of the next layer will be the the previous one
		hvInput = dvLayer;
		vNeuronValues.push_back(dvLayer);
	}

	for(int i = 0; i < hvInput.size(); i++) {
		std::cout<<"FW OUT: "<<hvInput[i]<<std::endl;
	}

	return vNeuronValues;
}

std::vector<ANN::Matrix>
hostBPPropagateBW(	std::vector<ANN::Matrix> &vEdgeMatricesO,
					std::vector<ANN::Matrix> &vEdgeMatricesI,
					std::vector<std::vector<float> > &vErrors,
					const std::vector<thrust::device_vector<float> > &vNeuronValues,
					const float &fLearningRate, 						// learning rate
					const ANN::TransfFunction &function) 				// deviation of transfer function
{
	// Create matrix for momentums
	std::vector<ANN::Matrix> vEdgeMomentums;
	for(unsigned int i = 0; i < vEdgeMatricesI.size(); i++) {
		ANN::Matrix matrix(vEdgeMatricesI.at(i).getW(), vEdgeMatricesI.at(i).getH(), 0.f);
		vEdgeMomentums.push_back(matrix);
	}
	
	thrust::device_vector	<float> 			dvErrors	(vErrors.back().begin(), vErrors.back().end());
	thrust::device_vector	<float> 			hvErrors;
	std::vector<thrust::device_vector<float> > 	vErrorDeltas(1, dvErrors);
	
	unsigned int iWidth 	= 0;
	unsigned int iHeight 	= 0;
	// calc it for the rest of the layers

	for(int i = vEdgeMatricesO.size()-1; i >= 0; i--) {					// All layers except output!
		iWidth 		= vEdgeMatricesO.at(i).getW();						// Nr. of neurons in next layer
		iHeight 	= vEdgeMatricesO.at(i).getH();						// Nr. of neurons in this layer

		std::vector<float> vCurrentErr = vErrors.at(i);
		dvErrors 	= thrust::device_vector<float>(vCurrentErr.begin(), vCurrentErr.end() );
		hvErrors 	= thrust::host_vector<float>(vErrorDeltas.front().begin(), vErrorDeltas.front().end() );
		
		if(hvErrors.size() == iHeight) {
			// Calculate the result of the current layer
			for(unsigned int y = 0; y < iHeight; y++) {	
				// Y <- A * X + Y
				thrust::transform(
				vEdgeMatricesO.at(i).getRowBegin(y), 					// X
				vEdgeMatricesO.at(i).getRowEnd(y), 						// X
				dvErrors.begin(), 										// Y
				dvErrors.begin(), 										// Y
				saxpy_functor(hvErrors[y]) ); 							// A
			}
			
			// Run values through transfer function
			thrust::transform(
					dvErrors.begin(),
					dvErrors.end(),
					dvErrors.begin(),
					unaryTransferFcn_functor(function) );

			// save in vector
			thrust::copy(dvErrors.begin(), dvErrors.end(), vErrors.at(i).begin());
			vErrorDeltas.insert(vErrorDeltas.begin(), dvErrors);
		}
	}
	
	// TODO GUESS THE PROBLEM IS HERE SOMEWHERE
	for(int i = vEdgeMatricesI.size()-1; i >= 0 && vNeuronValues.size() > 0; i--) { // All layers except output!
		iWidth 		= vEdgeMatricesI.at(i).getW();						// Nr. of neurons in next layer
		iHeight 	= vEdgeMatricesI.at(i).getH();						// Nr. of neurons in this layer

		//std::cout<<"iWidth: \t"<<iWidth<<std::endl;
		//std::cout<<"iHeight: \t"<<iHeight<<std::endl;

		// Calculate the result of the current layer
		for(unsigned int x = 0; x < iHeight; x++) {
			thrust::device_vector<float> dvResult(iWidth, 0.f);
			thrust::device_vector<float> dvMomentums(iWidth, 0.f);

			thrust::transform(
					vErrorDeltas.at(i+1).begin(), 							// X
					vErrorDeltas.at(i+1).end(), 							// X
					dvMomentums.begin(), 									// Y
					sax_functor(fLearningRate*vNeuronValues.at(i)[x]) ); 	// A

			for(int k = 0; k < iWidth; k++) {
				//std::cout<<"dvMomentums: "<<dvMomentums[k]<<std::endl;
			}

			for(int k = 0; k < iWidth; k++) {
				//std::cout<<"vEdgeMatricesO: "<<vEdgeMatricesI.at(i).getRowBegin(x)[k]<<std::endl;
			}

			thrust::transform(
					dvMomentums.begin(), 									// Y
					dvMomentums.end(), 										// Y
					vEdgeMatricesI.at(i).getRowBegin(x), 					// X
					vEdgeMatricesI.at(i).getRowBegin(x),//dvResult.begin(), 					// X
					thrust::plus<float>() );

			for(int k = 0; k < iWidth; k++) {
				//std::cout<<"Plus Result: "<<dvResult[k]<<std::endl;
				//std::cout<<"Plus Result: "<<vEdgeMatricesI.at(i).getRowBegin(x)[k]<<std::endl;
			}
		}
	}
	
	return vEdgeMomentums;
}

#endif
