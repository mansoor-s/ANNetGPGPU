/*
 * ANBPNetGPU.h
 *
 *  Created on: 14.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#ifndef ANBPNETGPU_H_
#define ANBPNETGPU_H_

#include <ANBPNet.h>
#include <gpgpu/ANKernels.h>
#include <gpgpu/ANMatrix.h>


namespace ANN {

class BPNetGPU: public ANN::BPNet {
private:
	std::vector<ANN::Matrix> m_vEdgeMatricesI;
	std::vector<ANN::Matrix> m_vMomentums;
	std::vector<ANN::Matrix> m_vBiasEdges;
	std::vector<thrust::device_vector<float> > m_vNeuronVals;
	std::vector<thrust::device_vector<float> > m_dvOutDeltas;

	void RefreshNeurons();
	void UpdateNeurons();		// only output layer (faster)


	void RefreshEdges();

	std::vector<float> GetCurrentInput();
	void GetEdgeMatrices();

	void GetErrorDeltas();
	void UpdateErrorDeltas(); 	// only output layer (faster)

public:
	BPNetGPU();
	virtual ~BPNetGPU();

	virtual float SetOutput(const std::vector<float> &vOutArray);
//	virtual float SetOutput(const std::vector<float> &outputArray, const unsigned int &layerID);
//	virtual float SetOutput(float *pOutArray, const unsigned int &size, const unsigned int &layerID);

	virtual void PropagateFW();
	virtual void PropagateBW();
	virtual std::vector<float> TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress);
};

}

#endif /* ANBPNETGPU_H_ */
