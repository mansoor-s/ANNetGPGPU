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
	std::vector<ANN::Matrix> m_vEdgeMatrices;

public:
	BPNetGPU();
	virtual ~BPNetGPU();

	virtual void PropagateFW();
	virtual void PropagateBW();
	virtual std::vector<float> TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress);
};

}

#endif /* ANBPNETGPU_H_ */
