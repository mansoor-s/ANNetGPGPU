/*
 * AbsNeuron.cpp
 *
 *  Created on: 01.09.2010
 *      Author: dgrat
 */

#include <iostream>
#include <stdio.h>
#include <cassert>
//own classes
#include <math/ANFunctions.h>
#include <math/ANRandom.h>
#include <basic/ANEdge.h>
#include <basic/ANAbsNeuron.h>
#include <ANBPLayer.h>
#include <containers/ANTrainingSet.h>

using namespace ANN;


AbsNeuron::AbsNeuron(AbsLayer *parentLayer) : m_pParentLayer(parentLayer) {
	/*
	 * Weise dem Neuron eine Zufallszahl zwischen 0 und 1 zu
	 * Genauigkeit liegt bei 4 Nachkommastellen
	 */
	m_fValue = RandFloat(-0.5f, 0.5f);

	m_fErrorDelta = 0;
	m_pBias = NULL;
}

AbsNeuron::AbsNeuron(const AbsNeuron *pNeuron) {
	float fErrorDelta 	= pNeuron->GetErrorDelta();
	float fValue 		= pNeuron->GetValue();
	float iID 			= pNeuron->GetID();

	this->SetErrorDelta(fErrorDelta);
	this->SetValue(fValue);
	this->SetID(iID);
}

AbsNeuron::~AbsNeuron() {
	m_lIncomingConnections.clear();
	m_lOutgoingConnections.clear();
}

void AbsNeuron::EraseAllEdges() {
	// TODO
/*
	for(int i = 0; i < m_lIncomingConnections.size(); i++) {
		if(m_lIncomingConnections[i] == NULL)
			continue;
		else {
			delete m_lIncomingConnections[i];
			m_lIncomingConnections[i] = NULL;
		}
	}
	for(int i = 0; i < m_lOutgoingConnections.size(); i++) {
		if(m_lOutgoingConnections[i] == NULL)
			continue;
		else {
			delete m_lOutgoingConnections[i];
			m_lOutgoingConnections[i] = NULL;
		}
	}
*/
	m_lIncomingConnections.clear();
	m_lOutgoingConnections.clear();
}

void AbsNeuron::AddConO(Edge *Edge) {
	m_lOutgoingConnections.push_back(Edge);
}

void AbsNeuron::AddConI(Edge *Edge) {
	m_lIncomingConnections.push_back(Edge);
}

void AbsNeuron::SetConO(Edge *Edge, const unsigned int iID) {
	m_lOutgoingConnections[iID] = Edge;
}

void AbsNeuron::SetConI(Edge *Edge, const unsigned int iID) {
	m_lIncomingConnections[iID] = Edge;
}

unsigned int AbsNeuron::GetID() const {
	return m_iNeuronID;
}

void AbsNeuron::SetID(const int ID) {
	m_iNeuronID = ID;
}

std::vector<Edge*> AbsNeuron::GetConsI() const{
	return m_lIncomingConnections;
}
std::vector<Edge*> AbsNeuron::GetConsO() const{
	return m_lOutgoingConnections;
}

Edge* AbsNeuron::GetConI(const unsigned int &pos) const {
	return m_lIncomingConnections.at(pos);
}

Edge* AbsNeuron::GetConO(const unsigned int &pos) const {
	return m_lOutgoingConnections.at(pos);
}


void AbsNeuron::SetValue(const float &value) {
	m_fValue = value;
}

void AbsNeuron::SetErrorDelta(const float &value)
{
	m_fErrorDelta = value;
}

void AbsNeuron::SetBiasEdge(Edge *Edge) {
	m_pBias = Edge;
}

const float &AbsNeuron::GetValue() const {
	return m_fValue;
}

const std::vector<float> AbsNeuron::GetPosition() const {
	return m_vPosition;
}

void AbsNeuron::SetPosition(const std::vector<float> &vPos) {
	m_vPosition = vPos;
}

const float &AbsNeuron::GetErrorDelta() const {
	return m_fErrorDelta;
}

Edge *AbsNeuron::GetBiasEdge() const {
	return m_pBias;
}

AbsLayer *AbsNeuron::GetParent() const {
	return m_pParentLayer;
}

void AbsNeuron::SetTransfFunction (const TransfFunction *pFCN) {
	this->m_ActFunction = pFCN;
}


const TransfFunction *AbsNeuron::GetTransfFunction() const {
	return (m_ActFunction);
}

AbsNeuron::operator float() const {
	return GetValue();
}

namespace ANN {
	/*
	 * AUSGABEOPERATOR
	 * OSTREAM
	 */
	std::ostream& operator << (std::ostream &os, AbsNeuron &op)
	{
	//	os << "Data of Neuron: " 									<< std::endl;
		os << "Value: \t" 		<< op.GetValue() 				<< std::endl;
	//    os << "Error delta: \t" << op.GetErrorDelta() 				<< std::endl;
	//    os << "Connections of Neuron:" 								<< std::endl;
	//    os << "Incoming: " 		<< op.GetConnectionsIn().size() 	<< std::endl;
	//    os << "Outgoing: " 		<< op.GetConnectionsOut().size() 	<< std::endl;
		return os;     // Ref. auf Stream
	}

	std::ostream& operator << (std::ostream &os, AbsNeuron *op)
	{
	//	os << "Data of Neuron: " 									<< std::endl;
		os << "Value: \t" 		<< op->GetValue() 			<< std::endl;
	//    os << "Delta: \t" << op->GetErrorDelta() 				<< std::endl;
	//    os << "Connections of Neuron:" 								<< std::endl;
	//    os << "Incoming: " 		<< op->GetConnectionsIn().size() 	<< std::endl;
	//    os << "Outgoing: " 		<< op->GetConnectionsOut().size() 	<< std::endl;
		return os;     // Ref. auf Stream
	}

	/*STATIC:*/
	void Connect(AbsNeuron *pSrcNeuron, AbsNeuron *pDstNeuron, const bool &bAdaptState) {
		Edge *pCurEdge = new Edge(pSrcNeuron, pDstNeuron);

		pCurEdge->SetAdaptationState(bAdaptState);
		pSrcNeuron->AddConO(pCurEdge);				// Edge beiden zuweisen
		pDstNeuron->AddConI(pCurEdge);
	}

	void Connect(AbsNeuron *pSrcNeuron, AbsNeuron *pDstNeuron, const float &fVal, const float &fMomentum, const bool &bAdaptState) {
		Edge *pCurEdge = new Edge(pSrcNeuron, pDstNeuron, fVal, fMomentum, bAdaptState);

		pSrcNeuron->AddConO(pCurEdge);				// Edge beiden zuweisen
		pDstNeuron->AddConI(pCurEdge);
	}

	void Connect(AbsNeuron *pSrcNeuron, AbsLayer *pDestLayer, const bool &bAdaptState) {
		unsigned int iSize 		= pDestLayer->GetNeurons().size();
		unsigned int iProgCount = 1;

		for(int j = 0; j < static_cast<int>(iSize); j++) {
			// Output
			if(((j+1) / (iSize/10)) == iProgCount && (j+1) % (iSize/10) == 0) {
				std::cout<<"Building connections.. Progress: "<<iProgCount*10.f<<"%/Step="<<j+1<<std::endl;
				iProgCount++;
			}
			// Work job
			Connect(pSrcNeuron, pDestLayer->GetNeuron(j), bAdaptState);
		}
	}

	void Connect(AbsNeuron *pSrcNeuron, AbsLayer *pDestLayer, const std::vector<float> &vValues, const std::vector<float> &vMomentums, const bool &bAdaptState) {
		unsigned int iSize 		= pDestLayer->GetNeurons().size();
		unsigned int iProgCount = 1;

		for(int j = 0; j < static_cast<int>(iSize); j++) {
			// Output
			if(((j+1) / (iSize/10)) == iProgCount && (j+1) % (iSize/10) == 0) {
				std::cout<<"Building connections.. Progress: "<<iProgCount*10.f<<"%/Step="<<j+1<<std::endl;
				iProgCount++;
			}
			// Work job
			Connect(pSrcNeuron, pDestLayer->GetNeuron(j), vValues[j], vMomentums[j], bAdaptState);
		}
	}
}
