/*
 * Edge.cpp
 *
 *  Created on: 30.05.2009
 *      Author: Xerces
 */

#include <iostream>
#include <cassert>
//own classes
#include <math/ANRandom.h>
#include <basic/ANEdge.h>
#include <basic/ANAbsNeuron.h>

using namespace ANN;


Edge::Edge(Edge *pEdge) {
	float 	fValue 		= pEdge->GetValue();
	float 	fMomentum 	= pEdge->GetMomentum();
	bool 	bAdapt 		= pEdge->GetAdaptationState();

	this->SetValue(fValue);
	this->SetMomentum(fMomentum);
	this->SetAdaptationState(bAdapt);

}

Edge::Edge(AbsNeuron *first, AbsNeuron *second) {
	assert(first);
	assert(second);

	m_pNeuronFirst 		= first;
	m_pNeuronSecond 	= second;

	m_fWeight 			= RandFloat(-0.5f, 0.5f);
	m_bAllowAdaptation 	= true;
	m_fMomentum 		= 0.f;
}

Edge::Edge(AbsNeuron *first, AbsNeuron *second, float fValue, float fMomentum, bool bAdapt) {
	assert(first);
	assert(second);

	m_pNeuronFirst 		= first;
	m_pNeuronSecond 	= second;

	m_fWeight 			= fValue;
	m_bAllowAdaptation 	= bAdapt;
	m_fMomentum 		= fMomentum;
}

AbsNeuron *Edge::GetDestination(AbsNeuron *source) const {
	assert(source);

	if(m_pNeuronFirst != source) {
		return m_pNeuronFirst;
	}
	else if(m_pNeuronSecond != source) {
		return m_pNeuronSecond;
	}
	else if(m_pNeuronFirst != source && m_pNeuronSecond != source) {
		std::cout<<"FEHLER: Quellneuron gehört nicht zur Kette"<<std::endl;
		return NULL;
	}
	else {
		std::cout<<"FEHLER: Edge enthält zwei gleiche Neuronen"<<std::endl;
		return NULL;
	}
}

int Edge::GetDestinationID(AbsNeuron *pSource) const {
	return Edge::GetDestination(pSource)->GetID();
}

const float &Edge::GetValue() const {
	return m_fWeight;
}

void Edge::SetValue(float fValue) {
	m_fWeight = fValue;
}

bool Edge::GetAdaptationState() const {
	return m_bAllowAdaptation;
}

void Edge::SetAdaptationState(const bool &adapt)	{
	m_bAllowAdaptation = adapt;
}

const float &Edge::GetMomentum() const {
	return m_fMomentum;
}

void Edge::SetMomentum(float fValue) {
	m_fMomentum = fValue;
}

Edge::operator float() const {
	return GetValue();
}
