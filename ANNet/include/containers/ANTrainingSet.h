/*
 * ANTrainingSet.h
 *
 *  Created on: 22.01.2011
 *      Author: dgrat
 */

#ifndef TRAININGDATA_H_
#define TRAININGDATA_H_

#include <utility>
#include <vector>

namespace ANN {


/**
 * \brief Storage of simple input/output samples usable for training.
 *
 * Data must get converted to simple float arrays to get used with this storage format.
 *
 * @author Daniel "dgrat" Frenzel
 */

class TrainingSet {
private:
	std::vector<std::vector<float> > m_vInputList;
	std::vector<std::vector<float> > m_vOutputList;

	// same like vectors, saved like a 2d array
	float *m_pInputList;	// for opencl implementation
	float *m_pOutputList;	// for opencl implementation
	unsigned int m_iOutW;
	unsigned int m_iOutH;
	unsigned int m_iInpW;
	unsigned int m_iInpH;

public:
	TrainingSet();
	~TrainingSet();

	void AddInput(const std::vector<float> &vIn);
	void AddOutput(const std::vector<float> &vOut);
	void AddInput(float *pIn, const unsigned int &iSize);
	void AddOutput(float *pOut, const unsigned int &iSize);

	unsigned int GetNrElements() const;

	std::vector<float> GetInput(const unsigned int &iID) const;
	std::vector<float> GetOutput(const unsigned int &iID) const;

	void Clear();

	// GPGPU implementation
	void CreateArrays();

	float *GetIArray();
	float *GetOArray();

	int GetIArraySize() const;
	int GetOArraySize() const;
};

}

#endif /* TRAININGDATA_H_ */
