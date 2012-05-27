/*
 * BPLayer.cpp
 *
 *  Created on: 02.06.2009
 *      Author: Xerces
 */

#include <cassert>
//own classes
#include <math/ANFunctions.h>
#include <basic/ANEdge.h>
#include <basic/ANAbsNeuron.h>
#include <ANBPNeuron.h>
#include <ANBPLayer.h>

using namespace ANN;


BPLayer::BPLayer() {
	m_pBiasNeuron = NULL;
}

BPLayer::BPLayer(const BPLayer *pLayer) {
	int iNumber 			= pLayer->GetNeurons().size();
	LayerTypeFlag fType 	= pLayer->GetFlag();
	m_pBiasNeuron 			= NULL;

	Resize(iNumber);
	SetFlag(fType);
}

BPLayer::BPLayer(const unsigned int &iNumber, LayerTypeFlag fType) {
	Resize(iNumber);
	m_pBiasNeuron = NULL;
	SetFlag(fType);
}

BPLayer::~BPLayer() {
	if(m_pBiasNeuron) {
		delete m_pBiasNeuron;
	}
}

void BPLayer::Resize(const unsigned int &iSize) {
	EraseAll();
	for(unsigned int i = 0; i < iSize; i++) {
		AbsNeuron *pNeuron = new BPNeuron(this);
		pNeuron->SetID(i);
		m_lNeurons.push_back(pNeuron);
	}
}

void BPLayer::ConnectLayer(AbsLayer *pDestLayer, const bool &bAllowAdapt) {
	AbsNeuron *pSrcNeuron;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "destLayer"
	 */
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		pSrcNeuron = m_lNeurons[i];
		Connect(pSrcNeuron, pDestLayer, bAllowAdapt);
	}

	if(m_pBiasNeuron) {
		Connect(m_pBiasNeuron, pDestLayer, true);
	}
}

void BPLayer::ConnectLayer(
		AbsLayer *pDestLayer,
		std::vector<std::vector<int> > Connections,
		const bool bAllowAdapt)
{
	AbsNeuron *pSrcNeuron;

	assert( Connections.size() != m_lNeurons.size() );
	for(unsigned int i = 0; i < Connections.size(); i++) {
		std::vector<int> subArray = Connections.at(i);
		pSrcNeuron = GetNeuron(i);
		assert(i != pSrcNeuron->GetID() );

		for(unsigned int j = 0; j < subArray.size(); j++) {
			assert( j < pDestLayer->GetNeurons().size() );
			AbsNeuron *pDestNeuron = pDestLayer->GetNeuron(j);
			assert( j < pDestNeuron->GetID() );
			Connect(pSrcNeuron, pDestNeuron, bAllowAdapt);
		}
	}

	if(m_pBiasNeuron) {
		Connect(m_pBiasNeuron, pDestLayer, true);
	}
}

BPNeuron *BPLayer::GetBiasNeuron() const {
	return m_pBiasNeuron;
}

void BPLayer::SetFlag(const LayerTypeFlag &fType) {
	m_fTypeFlag = fType;
	if( (m_fTypeFlag & ANBiasNeuron) && m_pBiasNeuron == NULL ) {
		m_pBiasNeuron = new BPNeuron(this);
		m_pBiasNeuron->SetValue(1.0f);
	}
}

void BPLayer::AddFlag(const LayerTypeFlag &fType) {
	if( !(m_fTypeFlag & fType) )
	m_fTypeFlag |= fType;
	if( (m_fTypeFlag & ANBiasNeuron) && m_pBiasNeuron == NULL ) {
		m_pBiasNeuron = new BPNeuron(this);
		m_pBiasNeuron->SetValue(1.0f);
	}
}

void BPLayer::SetLearningRate(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetLearningRate(fVal);
	}
}

void BPLayer::SetMomentum(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetMomentum(fVal);
	}
}

void BPLayer::SetWeightDecay(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetWeightDecay(fVal);
	}
}

/*
 * AUSGABEOPERATOR
 * OSTREAM
 */
std::ostream& operator << (std::ostream &os, BPLayer &op)
{
	if(op.GetBiasNeuron() != 0)
	os << "Bias neuron: \t" << op.GetBiasNeuron()->GetValue() 	<< std::endl;
    os << "Nr. neurons: \t" << op.GetNeurons().size() 					<< std::endl;
    return os;     // Ref. auf Stream
}

std::ostream& operator << (std::ostream &os, BPLayer *op)
{
	if(op->GetBiasNeuron() != 0)
	os << "Bias neuron: \t" << op->GetBiasNeuron()->GetValue()	<< std::endl;
    os << "Nr. neurons: \t" << op->GetNeurons().size() 					<< std::endl;
    return os;     // Ref. auf Stream
}

