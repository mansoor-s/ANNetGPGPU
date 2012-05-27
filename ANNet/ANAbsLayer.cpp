/*
 * AbsLayer.cpp
 *
 *  Created on: 21.02.2011
 *      Author: dgrat
 */

#include <cassert>
//own classes
#include <math/ANFunctions.h>
#include <basic/ANEdge.h>
#include <basic/ANAbsNeuron.h>
#include <basic/ANAbsLayer.h>

using namespace ANN;


AbsLayer::AbsLayer() {

}
/*
AbsLayer::AbsLayer(const unsigned int &iNumber, int iShiftID) {
	Resize(iNumber, iShiftID);
}
*/
AbsLayer::~AbsLayer() {
	EraseAll();
}

void AbsLayer::SetID(const int &iID) {
	m_iID = iID;
}

int AbsLayer::GetID() const {
	return m_iID;
}

const std::vector<AbsNeuron *> &AbsLayer::GetNeurons() const {
	return m_lNeurons;
}

void AbsLayer::EraseAllEdges() {
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		m_lNeurons[i]->EraseAllEdges();
	}
}

void AbsLayer::EraseAll() {
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		m_lNeurons[i]->EraseAllEdges();
		delete m_lNeurons[i];
	}
	m_lNeurons.clear();
}

AbsNeuron *AbsLayer::GetNeuron(const unsigned int &iID) const {
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		if(m_lNeurons.at(i)->GetID() == iID)
			return m_lNeurons.at(i);
	}
	return NULL;
}

void AbsLayer::SetNetFunction(const Function *pFunction) {
	assert( pFunction != 0 );
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		m_lNeurons[j]->SetNetFunction(pFunction);
	}
}

void AbsLayer::SetFlag(const LayerTypeFlag &fType) {
	m_fTypeFlag = fType;
}

void AbsLayer::AddFlag(const LayerTypeFlag &fType) {
	if( !(m_fTypeFlag & fType) )
	m_fTypeFlag |= fType;
}

LayerTypeFlag AbsLayer::GetFlag() const {
	return m_fTypeFlag;
}

/*FRIEND:*/
void SetEdgesToValue(AbsLayer *pSrcLayer, AbsLayer *pDestLayer, const float &fVal, const bool &bAdaptState) {
	AbsNeuron	*pCurNeuron;
	Edge 		*pCurEdge;
	for(unsigned int i = 0; i < pSrcLayer->GetNeurons().size(); i++) {
		pCurNeuron = pSrcLayer->GetNeurons().at(i);
		for(unsigned int j = 0; j < pCurNeuron->GetConsO().size(); j++) {
			pCurEdge = pCurNeuron->GetConO(j);
			// outgoing edge is connected with pDestLayer ..
			if(pCurEdge->GetDestination(pCurNeuron)->GetParent() == pDestLayer) {
				// .. d adapt only these edges
				pCurEdge->SetValue( fVal );
				pCurEdge->SetAdaptationState( bAdaptState );
			}
		}
	}
}

F2DArray AbsLayer::ExpEdgesIn() const {
	unsigned int iHeight = m_lNeurons.at(0)->GetConsI().size();
	unsigned int iWidth = m_lNeurons.size();

	assert(iWidth > 0 && iHeight > 0);

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetConI(y)->GetValue();
			//std::cout << "ExpEdgesIn() - weight: " << m_lNeurons.at(x)->GetConI(y)->GetValue() << std::endl;
		}
	}
	return vRes;
}

F2DArray AbsLayer::ExpEdgesOut() const {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetConsO().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iWidth > 0 && iHeight > 0);

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetConO(y)->GetValue();
		}
	}
	return vRes;
}

void AbsLayer::ImpEdgesIn(const F2DArray &mat) {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetConsI().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_lNeurons.at(x)->GetConI(y)->SetValue(mat[y][x]);
		}
	}
}

void AbsLayer::ImpEdgesOut(const F2DArray &mat) {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetConsO().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_lNeurons.at(x)->GetConO(y)->SetValue(mat[y][x]);
		}
	}
}

F2DArray AbsLayer::ExpPositions() const {
	unsigned int iHeight = m_lNeurons.at(0)->GetPosition().size();
	unsigned int iWidth = m_lNeurons.size();

	assert(iWidth > 0 && iHeight > 0);

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetPosition().at(y);
		}
	}
	return vRes;
}

void AbsLayer::ImpPositions(const F2DArray &f2dPos) {
	unsigned int iHeight = f2dPos.GetH();
	unsigned int iWidth = f2dPos.GetW();

	assert(iWidth == m_lNeurons.size() );

	#pragma omp parallel for
	for(int x = 0; x < static_cast<int>(iWidth); x++) {
		std::vector<float> vPos(iHeight);
		for(unsigned int y = 0; y < iHeight; y++) {
			vPos[y] = f2dPos.m_pArray[y*iWidth+x];
		}
		m_lNeurons.at(x)->SetPosition(vPos);
	}
}
