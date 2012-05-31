/*
 * AbsNet.cpp
 *
 *  Created on: 16.02.2011
 *      Author: dgrat
 */


#include <iostream>
#include <cassert>
#include <omp.h>
//own classes
#include <math/ANRandom.h>
#include <math/ANFunctions.h>
#include <containers/ANTrainingSet.h>
#include <containers/ANConTable.h>
#include <basic/ANEdge.h>
#include <basic/ANAbsNeuron.h>
#include <basic/ANAbsNet.h>
#include <ANBPLayer.h>

using namespace ANN;


AbsNet::AbsNet() : Importer(this),  Exporter(this) {
	m_fLearningRate = 0.0f;
	m_fMomentum 	= 0.f;
	m_fWeightDecay 	= 0.f;
	m_ActFunction 	= NULL;
	m_pTrainingData = NULL;

	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;

	m_fTypeFlag 	= ANNetUndefined;

	// Init time for rdom numbers
	INIT_TIME
}

AbsNet::AbsNet(AbsNet *pNet) : Importer(this),  Exporter(this) {
	assert( pNet != NULL );
	*this = *pNet;
}

AbsNet::~AbsNet() {
	this->EraseAll();
}

void AbsNet::SetFlag(const NetTypeFlag &fType) {
	m_fTypeFlag = fType;
}

void AbsNet::AddFlag(const NetTypeFlag &fType) {
	if( !(m_fTypeFlag & fType) )
		m_fTypeFlag |= fType;
}

LayerTypeFlag AbsNet::GetFlag() const {
	return m_fTypeFlag;
}

void AbsNet::EraseAll() {
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>( m_lLayers.size() ); i++) {
		m_lLayers.at(i)->EraseAll();
	}
	m_lLayers.clear();
}

std::vector<float> AbsNet::TrainFromData(const unsigned int &iCycles, const float &fTolerance) {
	std::vector<float> pErrors;

	if(m_pTrainingData == NULL)
		return pErrors;

//	float fStepSize 	= 0.01f;
//	float fLastError 	= 0.f;
	float fCurError 	= 0.f;

	for(unsigned int j = 0; j < iCycles; j++) {
		if(fCurError < fTolerance && j > 0) {
			std::cout<<"Break after: "<<j<<" cycles.\nLast total error: "<<fCurError<<"\nLearning rate: "<<m_fLearningRate<<std::endl;
			return pErrors;
		}
/*
		if(fCurError >= fLastError)
			SetLearningRate(m_fLearningRate + fStepSize);
		else {
			if(m_fLearningRate - fStepSize > 0)
				SetLearningRate(m_fLearningRate - fStepSize);
		}
		fLastError 	= fCurError;
*/
		fCurError 	= 0.f;
		for( unsigned int i = 0; i < m_pTrainingData->GetNrElements(); i++ ) {
			SetInput( m_pTrainingData->GetInput(i) );
			fCurError += SetOutput( m_pTrainingData->GetOutput(i) );
			PropagateBW();
		}
		pErrors.push_back(fCurError);
	}
	std::cout<<"Break\nLast total error: "<<fCurError<<"\nLearning rate: "<<m_fLearningRate<<std::endl;
	return pErrors;
}

void AbsNet::AddLayer(AbsLayer *pLayer) {
	m_lLayers.push_back(pLayer);
	pLayer->SetID( m_lLayers.size()-1 );
}

std::vector<AbsLayer*> AbsNet::GetLayers() const {
	return m_lLayers;
}

AbsLayer* AbsNet::GetLayer(const unsigned int &iLayerID) const {
	return m_lLayers.at(iLayerID);
}

void AbsNet::SetInput(const std::vector<float> &inputArray) {
	assert( m_pIPLayer != NULL );
	assert( inputArray.size() <= m_pIPLayer->GetNeurons().size() );

	AbsNeuron *pCurNeuron;
	for(int i = 0; i < static_cast<int>( m_pIPLayer->GetNeurons().size() ); i++) {
		pCurNeuron = m_pIPLayer->GetNeuron(i);
		pCurNeuron->SetValue(inputArray[i]);
	}
}

void AbsNet::SetInput(const std::vector<float> &inputArray, const unsigned int &layerID) {
// TODO implement
//	assert( m_lLayers[layerID]->GetFlag() & LayerInput );
	assert( layerID < m_lLayers.size() );
	assert( inputArray.size() <= m_lLayers[layerID]->GetNeurons().size() );

	AbsNeuron *pCurNeuron;
	for(int i = 0; i < static_cast<int>( m_lLayers[layerID]->GetNeurons().size() ); i++) {
		pCurNeuron = m_lLayers[layerID]->GetNeuron(i);
		pCurNeuron->SetValue(inputArray[i]);
	}
}

void AbsNet::SetInput(float *inputArray, const unsigned int &size, const unsigned int &layerID) {
//	assert( m_lLayers[layerID]->GetFlag() & LayerInput );
	assert( layerID < m_lLayers.size() );
	assert( size <= m_lLayers[layerID]->GetNeurons().size() );

	AbsNeuron *pCurNeuron;
	for(int i = 0; i < static_cast<int>( m_lLayers[layerID]->GetNeurons().size() ); i++) {
		pCurNeuron = m_lLayers[layerID]->GetNeuron(i);
		pCurNeuron->SetValue(inputArray[i]);
	}
}

float AbsNet::SetOutput(const std::vector<float> &outputArray) {
	assert( m_pOPLayer != NULL );
	assert( outputArray.size() == m_pOPLayer->GetNeurons().size() );

	PropagateFW();

	float fError 		= 0.f;
	float fCurError 	= 0.f;
	AbsNeuron *pCurNeuron = NULL;
	for(unsigned int i = 0; i < m_pOPLayer->GetNeurons().size(); i++) {
		pCurNeuron = m_pOPLayer->GetNeuron(i);
		fCurError = outputArray[i] - pCurNeuron->GetValue();
		fError += pow( fCurError, 2 ) / 2.f;
		pCurNeuron->SetErrorDelta(fCurError);
	}
	return fError;
}

float AbsNet::SetOutput(const std::vector<float> &outputArray, const unsigned int &layerID) {
//	assert( m_lLayers[layerID]->GetFlag() & LayerOutput );
	assert( layerID < m_lLayers.size() );
	assert( outputArray.size() == m_lLayers[layerID]->GetNeurons().size() );

	PropagateFW();

	float fError 		= 0.f;
	float fCurError 	= 0.f;
	AbsNeuron *pCurNeuron = NULL;
	for(unsigned int i = 0; i < m_lLayers[layerID]->GetNeurons().size(); i++) {
		pCurNeuron = m_lLayers[layerID]->GetNeuron(i);
		fCurError = outputArray[i] - pCurNeuron->GetValue();
		fError += pow( fCurError, 2 ) / 2.f;
		pCurNeuron->SetErrorDelta(fCurError);
	}
	return fError;
}

float AbsNet::SetOutput(float *outputArray, const unsigned int &size, const unsigned int &layerID) {
//	assert( m_lLayers[layerID]->GetFlag() & LayerOutput );
	assert( layerID < m_lLayers.size() );
	assert( size == m_lLayers[layerID]->GetNeurons().size() );

	PropagateFW();

	float fError 		= 0.f;
	float fCurError 	= 0.f;
	AbsNeuron *pCurNeuron = NULL;
	for(unsigned int i = 0; i < m_lLayers[layerID]->GetNeurons().size(); i++) {
		pCurNeuron = m_lLayers[layerID]->GetNeuron(i);
		fCurError = outputArray[i] - pCurNeuron->GetValue();
		fError += pow( fCurError, 2 ) / 2.f;
		pCurNeuron->SetErrorDelta(fCurError);
	}
	return fError;
}

void AbsNet::SetTrainingSet(TrainingSet *pData) {
	if( pData != NULL ) {
		m_pTrainingData = pData;
	}
}

void AbsNet::SetTrainingSet(const TrainingSet &pData) {
	m_pTrainingData = (TrainingSet*)&pData;
}

TrainingSet *AbsNet::GetTrainingSet() const {
	return m_pTrainingData;
}

const AbsLayer *AbsNet::GetIPLayer() const {
	return m_pIPLayer;
}

const AbsLayer *AbsNet::GetOPLayer() const {
	return m_pOPLayer;
}

void AbsNet::SetIPLayer(const unsigned int iID) {
	assert (iID >= 0);
	assert (iID <= GetLayers().size() );

	m_pIPLayer = GetLayer(iID);
}

void AbsNet::SetOPLayer(const unsigned int iID) {
	assert (iID >= 0);
	assert (iID <= GetLayers().size() );

	m_pOPLayer = GetLayer(iID);
}

std::vector<float> AbsNet::GetOutput() {
	assert( m_pOPLayer != NULL );

	std::vector<float> vResult;
	for(unsigned int i = 0; i < GetOPLayer()->GetNeurons().size(); i++) {
		AbsNeuron *pCurNeuron = GetOPLayer()->GetNeuron(i);
		vResult.push_back(pCurNeuron->GetValue() );
	}

	return vResult;
}

void AbsNet::SetNetFunction(const TransfFunction *pFunction) {
	assert( pFunction != 0 );

	m_ActFunction = pFunction;
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		GetLayer(i)->SetNetFunction(m_ActFunction);
	}
}

const TransfFunction *AbsNet::GetNetFunction() const {
	return m_ActFunction;
}

/*
 * AUSGABEOPERATOR
 * OSTREAM
 */
namespace ANN {

	std::ostream& operator << (std::ostream &os, AbsNet &op) {
		assert( op.GetOPLayer() != NULL );

		/* if training data was set give out all samples */
		if( op.GetTrainingSet() != NULL ) {
			for( unsigned int i = 0; i < op.GetTrainingSet()->GetNrElements(); i++ ) {
				op.SetInput( op.GetTrainingSet()->GetInput(i) );
				op.PropagateFW();

				for(unsigned int j = 0; j < op.GetOPLayer()->GetNeurons().size(); j++) {
					AbsNeuron *pCurNeuron = op.GetOPLayer()->GetNeuron(j);
					std::cout << pCurNeuron;
				}
				std::cout << std::endl;
			}
		}
		else {
			for(unsigned int i = 0; i < op.GetOPLayer()->GetNeurons().size(); i++) {
				AbsNeuron *pCurNeuron = op.GetOPLayer()->GetNeuron(i);
				std::cout << pCurNeuron;
			}
		}
		return os;     // Ref. auf Stream
	}

	std::ostream& operator << (std::ostream &os, AbsNet *op) {
		assert( op->GetOPLayer() != NULL );

		/* if training data was set give out all samples */
		if( op->GetTrainingSet() != NULL ) {
			for( unsigned int i = 0; i < op->GetTrainingSet()->GetNrElements(); i++ ) {
				std::cout << "Output: "<< i << std::endl;
				op->SetInput( op->GetTrainingSet()->GetInput(i) );
				op->PropagateFW();

				for(unsigned int j = 0; j < op->GetOPLayer()->GetNeurons().size(); j++) {
					AbsNeuron *pCurNeuron = op->GetOPLayer()->GetNeuron(j);
					std::cout << pCurNeuron;
				}
				std::cout << std::endl;
			}
		}
		else {
			for(unsigned int i = 0; i < op->GetOPLayer()->GetNeurons().size(); i++) {
				AbsNeuron *pCurNeuron = op->GetOPLayer()->GetNeuron(i);
				std::cout << pCurNeuron;
			}
		}
		return os;     // Ref. auf Stream
	}

}
