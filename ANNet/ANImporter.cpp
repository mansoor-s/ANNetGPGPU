/*
 * Importer.cpp
 *
 *  Created on: 10.02.2011
 *      Author: dgrat
 */

#include <iostream>
#include <omp.h>
#include <bzlib.h>

#include <basic/ANAbsNet.h>
#include <basic/ANImporter.h>
#include <containers/ANConTable.h>
#include <ANBPLayer.h>
#include <ANBPNeuron.h>
#include <ANSOMLayer.h>
#include <ANSOMNeuron.h>
#include <ANHFLayer.h>
#include <ANHFNeuron.h>

using namespace ANN;


Importer::Importer(AbsNet *pNet) {
	m_pNet = pNet;
}

Importer::~Importer() {
}

// NEW LOADFUNCTION
void Importer::ImpFromFS(std::string path) {
	int iBZ2Error;

	ConTable Res;

	NetTypeFlag 	fNetType;
	LayerTypeFlag 	fLayerType;
	unsigned int iNmbDims 			= 0;
	unsigned int iNmbLayers 		= 0;	// zahl der Layer im Netz
	unsigned int iNmbOfNeurons 		= 0;	// zahl der Neuronen im Layer
	unsigned int iNmbOfConnects 	= 0;

	std::vector<float> vNeuronPos;

	float fEdgeValue 	= 0.0f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;
	int iSrcNeurID 		= -1;

	bool bHasBias = false;

	FILE *fin = fopen(path.c_str(), "rb");
	BZFILE* bz2in;
	bz2in = BZ2_bzReadOpen(&iBZ2Error, fin, 0, 0, NULL, 0);

	if (iBZ2Error != BZ_OK) {														// ABBRUCHBEDINGUNG
		std::cout<<"return: "<<"LoadNetwork()"<<std::endl;
		return;
	}
	std::cout<<"Load network.."<<std::endl;
	BZ2_bzRead( &iBZ2Error, bz2in, &fNetType, sizeof(int) );
	Res.NetType = fNetType;
	BZ2_bzRead( &iBZ2Error, bz2in, &iNmbLayers, sizeof(int) );
	Res.NrOfLayers = iNmbLayers;
	for(unsigned int i = 0; i < iNmbLayers; i++) {
		BZ2_bzRead( &iBZ2Error, bz2in, &bHasBias, sizeof(bool) );
		BZ2_bzRead( &iBZ2Error, bz2in, &fLayerType, sizeof(LayerTypeFlag) );
		BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfNeurons, sizeof(int) );
		Res.TypeOfLayer.push_back(fLayerType);
		Res.SizeOfLayer.push_back(iNmbOfNeurons);

		for(unsigned int j = 0; j < iNmbOfNeurons; j++) {
			BZ2_bzRead( &iBZ2Error, bz2in, &iSrcNeurID, sizeof(int) );
			/*
			 * Save positions of the neurons
			 * important for SOMs
			 */
			BZ2_bzRead( &iBZ2Error, bz2in, &iNmbDims, sizeof(int) );
			vNeuronPos.resize(iNmbDims);
			for(unsigned int k = 0; k < iNmbDims; k++) {
				BZ2_bzRead( &iBZ2Error, bz2in, &vNeuronPos[k], sizeof(float) );
			}
			NeurDescr 	cCurNeur;
			cCurNeur.m_iLayerID 	= i;
			cCurNeur.m_iNeurID 		= iSrcNeurID;
			cCurNeur.m_vPos 		= vNeuronPos;
			Res.Neurons.push_back(cCurNeur);
			/*
			 * Save data of connections
			 */
			BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfConnects, sizeof(int) );
			for(unsigned int k = 0; k < iNmbOfConnects; k++) {
				BZ2_bzRead( &iBZ2Error, bz2in, &iDstLayerID, sizeof(int) );
				BZ2_bzRead( &iBZ2Error, bz2in, &iDstNeurID, sizeof(int) );
				BZ2_bzRead( &iBZ2Error, bz2in, &fEdgeValue, sizeof(float) );
				ConDescr	cCurCon;
				cCurCon.m_fVal 			= fEdgeValue;
				cCurCon.m_iSrcNeurID 	= iSrcNeurID;
				cCurCon.m_iDstNeurID 	= iDstNeurID;
				cCurCon.m_iSrcLayerID 	= i;			// current array always equal to current index, so valid
				cCurCon.m_iDstLayerID 	= iDstLayerID;	// last chge
				Res.NeurCons.push_back(cCurCon);
			}
		}
		/*
		 * Only for back propagation networks
		 */
		if(bHasBias && fNetType == ANNetBP) {
			BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfConnects, sizeof(int) );
			for(unsigned int j = 0; j < iNmbOfConnects; j++) {
				BZ2_bzRead( &iBZ2Error, bz2in, &iDstLayerID, sizeof(int) );
				BZ2_bzRead( &iBZ2Error, bz2in, &iDstNeurID, sizeof(int) );
				BZ2_bzRead( &iBZ2Error, bz2in, &fEdgeValue, sizeof(float) );
				ConDescr	cCurCon;
				cCurCon.m_fVal 			= fEdgeValue;
				cCurCon.m_iDstNeurID 	= iDstNeurID;
				cCurCon.m_iSrcLayerID 	= i;
				cCurCon.m_iDstLayerID 	= iDstLayerID;
				Res.BiasCons.push_back(cCurCon);
			}
		}
	}

	CreateNet( Res );
	BZ2_bzReadClose ( &iBZ2Error, bz2in );
	fclose(fin);
}

void Importer::CreateNet(const ConTable &Net) {
	std::cout<<"Create net"<<std::endl;

	/*
	 * Initialisiere Variablen
	 */
	unsigned int iNmbLayers 	= Net.NrOfLayers;	// zahl der Layer im Netz
	unsigned int iNmbNeurons	= 0;

	unsigned int iDstNeurID 	= 0;
	unsigned int iSrcNeurID 	= 0;
	unsigned int iDstLayerID 	= 0;
	unsigned int iSrcLayerID 	= 0;

	float fEdgeValue			= 0.f;

	AbsLayer *pDstLayer 		= NULL;
	AbsLayer *pSrcLayer 		= NULL;
	AbsNeuron *pDstNeur 		= NULL;
	AbsNeuron *pSrcNeur 		= NULL;

	LayerTypeFlag fType 		= 0;
	/*
	 *	Lösche Netz
	 */
	m_pNet->EraseAll();
	m_pNet->SetFlag(Net.NetType);
	/*
	 * Erstelle die Netzschichten
	 */
	for(unsigned int i = 0; i < iNmbLayers; i++) {
		iNmbNeurons = Net.SizeOfLayer.at(i);
		fType 		= Net.TypeOfLayer.at(i);
		// Create layers
		if(Net.NetType == ANNetBP) {
			m_pNet->AddLayer( new BPLayer(iNmbNeurons, fType) );
		}
		else if(Net.NetType == ANNetSOM) {
			m_pNet->AddLayer( new SOMLayer(iNmbNeurons, fType) );
		}
		else if(Net.NetType == ANNetHopfield) {
			m_pNet->AddLayer( new HFLayer(iNmbNeurons, 1) );
		}
		// Set pointers to input and output layers
		if(fType == ANLayerInput) {
			m_pNet->SetIPLayer(i);
		}
		else if(fType == ANLayerOutput) {
			m_pNet->SetOPLayer(i);
		}
		else if(fType == (ANLayerInput | ANLayerOutput) ) {	// Hopfield networks
			m_pNet->SetIPLayer(i);
			m_pNet->SetOPLayer(i);
		}
	}
	/*
	 * Only for SOMs
	 */
	for(unsigned int i = 0; i < Net.Neurons.size(); i++) {
		int iLayerID 	= Net.Neurons.at(i).m_iLayerID;
		int iNeurID 	= Net.Neurons.at(i).m_iNeurID;
		std::vector<float> vPos = Net.Neurons.at(i).m_vPos;
		m_pNet->GetLayer(iLayerID)->GetNeuron(iNeurID)->SetPosition(vPos);
	}
	/*
	 * Basic information for ~all networks
	 */
	for(unsigned int i = 0; i < Net.NeurCons.size(); i++) {
		iDstNeurID = Net.NeurCons.at(i).m_iDstNeurID;
		iSrcNeurID = Net.NeurCons.at(i).m_iSrcNeurID;
		iDstLayerID = Net.NeurCons.at(i).m_iDstLayerID;
		iSrcLayerID = Net.NeurCons.at(i).m_iSrcLayerID;
		if(iDstNeurID < 0 || iSrcNeurID < 0 || iDstLayerID < 0 || iSrcLayerID < 0) {
			return;
		}
		else {
			fEdgeValue 	= Net.NeurCons.at(i).m_fVal;

			pDstLayer 	= ( m_pNet->GetLayer(iDstLayerID) );
			pSrcLayer 	= ( m_pNet->GetLayer(iSrcLayerID) );

			pDstNeur 	= pDstLayer->GetNeuron(iDstNeurID);
			pSrcNeur 	= pSrcLayer->GetNeuron(iSrcNeurID);
			Connect(pSrcNeur, pDstNeur, fEdgeValue, 0.f, true);
		}
	}
	/*
	 * Only for back propagation networks
	 */
	if(Net.NetType == ANNetBP) {
		for(unsigned int i = 0; i < Net.BiasCons.size(); i++) {
			iDstNeurID = Net.BiasCons.at(i).m_iDstNeurID;
			iDstLayerID = Net.BiasCons.at(i).m_iDstLayerID;
			iSrcLayerID = Net.BiasCons.at(i).m_iSrcLayerID;
			if(iDstNeurID < 0 || iDstLayerID < 0 || m_pNet->GetLayers().size() < iDstLayerID || m_pNet->GetLayers().size() < iSrcLayerID) {
				return;
			}
			else {
				fEdgeValue 	= Net.BiasCons.at(i).m_fVal;

				pDstLayer 	= ( (BPLayer*)m_pNet->GetLayer(iDstLayerID) );
				pSrcLayer 	= ( (BPLayer*)m_pNet->GetLayer(iSrcLayerID) );
				pSrcNeur 	= ( (BPLayer*)pSrcLayer)->GetBiasNeuron();

				pDstNeur 	= pDstLayer->GetNeuron(iDstNeurID);
				Connect(pSrcNeur, pDstNeur, fEdgeValue, 0.f, true);
			}
		}
	}
}
