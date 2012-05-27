/*
 * ANSOMNetGPU.h
 *
 *  Created on: 01.04.2012
 *      Author: dgrat
 */

#ifndef ANSOMNETGPU_H_
#define ANSOMNETGPU_H_

#include <ANSOMNet.h>
#include <gpgpu/ANKernels.h>
#include <gpgpu/ANMatrix.h>


namespace ANN {

class SOMNetGPU : public SOMNet {
private:
	ANN::Matrix m_EdgeMat;
	ANN::Matrix m_PosiMat;

public:
	SOMNetGPU();
	SOMNetGPU(AbsNet *pNet);
	virtual ~SOMNetGPU();

	/**
	 * Trains the network with given input until iCycles is reached.
	 * @param iCycles Maximum number of training cycles.
	 */
	virtual void Training(const unsigned int &iCycles = 1000);
};

}

#endif /* ANSOMNETGPU_H_ */
