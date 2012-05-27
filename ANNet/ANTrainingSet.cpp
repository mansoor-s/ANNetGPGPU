/*
 * TrainingData.cpp
 *
 *  Created on: 22.01.2011
 *      Author: dgrat
 */

#include <cassert>
#include <cstddef>
// own classes
#include <containers/ANTrainingSet.h>

using namespace ANN;


TrainingSet::TrainingSet() {
	m_pInputList = NULL;
	m_pOutputList = NULL;
}

TrainingSet::~TrainingSet() {
	if(m_pOutputList != NULL) {
		delete [] m_pOutputList;
	}
	if(m_pInputList != NULL) {
		delete [] m_pInputList;
	}

	Clear();
}

void TrainingSet::AddInput(const std::vector<float> &vIn) {
	m_vInputList.push_back(vIn);
}

void TrainingSet::AddOutput(const std::vector<float> &vOut) {
	m_vOutputList.push_back(vOut);
}

void TrainingSet::AddInput(float *pIn, const unsigned int &iSize) {
	std::vector<float> vIn;
	for(unsigned int i = 0; i < iSize; i++)
		vIn.push_back(pIn[i]);
	AddInput(vIn);
}

void TrainingSet::AddOutput(float *pOut, const unsigned int &iSize) {
	std::vector<float> vOut;
	for(unsigned int i = 0; i < iSize; i++)
		vOut.push_back(pOut[i]);
	AddOutput(vOut);
}

unsigned int TrainingSet::GetNrElements() const {
/*
	assert( m_vInputList.size() == m_vOutputList.size() );
	assert( m_vInputList.size() > 0 );
	assert( m_vOutputList.size() > 0 );
*/
	return m_vInputList.size();
}

std::vector<float> TrainingSet::GetInput(const unsigned int &iID) const {
	return m_vInputList.at(iID);
}

std::vector<float> TrainingSet::GetOutput(const unsigned int &iID) const {
	return m_vOutputList.at(iID);
}

void TrainingSet::Clear() {
	m_vInputList.clear(); m_vOutputList.clear();
}

// OPENCL
void TrainingSet::CreateArrays() {
	/*
	 * Alloc memory
	 */
	unsigned int iMaxW = 0;
	unsigned int iMaxH = m_vInputList.size();
	for(unsigned int i = 0; i < iMaxH; i++) {
		if( iMaxW < m_vInputList.at(i).size() )
			iMaxW = m_vInputList.at(i).size();
	}
	m_iInpW = iMaxW;
	m_iInpH = iMaxH;
	assert(m_iInpH*m_iInpW > 0);
	m_pInputList = new float [iMaxH*iMaxW];

	iMaxH = m_vOutputList.size();
	for(unsigned int i = 0; i < iMaxH; i++) {
		if( iMaxW < m_vOutputList.at(i).size() )
			iMaxW = m_vOutputList.at(i).size();
	}
	m_iOutW = iMaxW;
	m_iOutH = iMaxH;
	assert(m_iOutW*m_iOutH > 0);
	m_pOutputList = new float [iMaxH*iMaxW];

	/*
	 * Fill memory
	 */
	assert(m_vInputList.size() > m_iInpH);
	for(unsigned int y = 0; y < m_vInputList.size(); y++) {
		for(unsigned int x = 0; x < m_vInputList.at(y).size(); x++) {
			m_pInputList[y*m_iInpW+x] = m_vInputList[y][x];
		}
	}
	assert(m_vOutputList.size() > m_iOutH);
	for(unsigned int y = 0; y < m_vOutputList.size(); y++) {
		for(unsigned int x = 0; x < m_vOutputList.at(y).size(); x++) {
			m_pOutputList[y*m_iInpW+x] = m_vOutputList[y][x];
		}
	}
}

float *TrainingSet::GetIArray() {
	assert(m_pInputList != NULL);
	return m_pInputList;
}

float *TrainingSet::GetOArray() {
	assert(m_pOutputList != NULL);
	return m_pOutputList;
}

int TrainingSet::GetIArraySize() const {
	assert(m_iInpH*m_iInpW > 0);
	return (m_iInpH*m_iInpW);
}

int TrainingSet::GetOArraySize() const {
	assert(m_iOutW*m_iOutH > 0);
	return (m_iOutW*m_iOutH);
}
