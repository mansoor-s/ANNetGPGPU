/*
 * Exporter.cpp
 *
 *  Created on: 09.02.2011
 *      Author: dgrat
 */

#include <string>
#include <omp.h>
#include <bzlib.h>

// own classes
#include <containers/AN2DArray.h>
#include <containers/AN3DArray.h>
#include <basic/ANEdge.h>
#include <basic/ANAbsNet.h>
#include <basic/ANExporter.h>
#include <ANBPNeuron.h>
#include <ANBPLayer.h>

using namespace ANN;


Exporter::Exporter(AbsNet *pNet) {
	m_pNet 		= pNet;
	m_pNeurons 	= NULL;
	m_pDeltas 	= NULL;
	m_pWeights 	= NULL;
}

Exporter::~Exporter() {
}

void Exporter::ExpToFS(std::string path) const {
	int iBZ2Error;

	NetTypeFlag 	fNetType = m_pNet->GetFlag();
	LayerTypeFlag 	fLayerType;
	unsigned int iNmbDims 			= 0;
	unsigned int iNmbOfNeurons 		= 0;
	unsigned int iNmbOfConnects 	= 0;
	unsigned int iNmbOfLayers 		= m_pNet->GetLayers().size();

	float fEdgeValue 	= 0.0f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;
	int iSrcNeurID 		= -1;

	bool bHasBias = false;

	FILE *fout = fopen(path.c_str(), "wb");
	BZFILE* bz2out;
	bz2out = BZ2_bzWriteOpen(&iBZ2Error, fout, 9, 0, 0);

	if (iBZ2Error != BZ_OK) {
		std::cout<<"return: "<<"SaveNetwork()"<<std::endl;
		return;
	}
	std::cout<<"Save network.."<<std::endl;
	BZ2_bzWrite( &iBZ2Error, bz2out, &fNetType, sizeof(int) );
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfLayers, sizeof(int) );
	for(unsigned int i = 0; i < iNmbOfLayers; i++) {
		BPLayer *pCurLayer = ( (BPLayer*)m_pNet->GetLayer(i) );					// BPLayer has greatest subset of functionality yet so i have to use it here
		fLayerType = pCurLayer->GetFlag();
		iNmbOfNeurons = pCurLayer->GetNeurons().size();
		if(pCurLayer->GetBiasNeuron() != NULL) {
			bHasBias = true;
		}
		else bHasBias = false;
		BZ2_bzWrite( &iBZ2Error, bz2out, &bHasBias, sizeof(bool) );				// Has layer bias neuron
		BZ2_bzWrite( &iBZ2Error, bz2out, &fLayerType, sizeof(LayerTypeFlag) );	// Type of layer
		BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfNeurons, sizeof(int) );			// Number of neuron in this layer (except bias)
		for(unsigned int j = 0; j < iNmbOfNeurons; j++) {
			AbsNeuron *pCurNeur = pCurLayer->GetNeuron(j);
			iSrcNeurID = pCurNeur->GetID();
			iNmbOfConnects = pCurNeur->GetConsO().size();
			BZ2_bzWrite( &iBZ2Error, bz2out, &iSrcNeurID, sizeof(int) );
			/*
			 * Save positions of the neurons
			 * important for SOMs
			 */
			iNmbDims = pCurNeur->GetPosition().size();
			BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbDims, sizeof(int) );
			for(unsigned int k = 0; k < iNmbDims; k++) {
				float fPos = pCurNeur->GetPosition().at(k);
				BZ2_bzWrite( &iBZ2Error, bz2out, &fPos, sizeof(float) );
			}
			/*
			 * Save data of connections
			 */
			BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfConnects, sizeof(int) );
			for(unsigned int k = 0; k < iNmbOfConnects; k++) {
				Edge *pCurEdge = pCurNeur->GetConO(k);
				iDstLayerID = pCurEdge->GetDestination(pCurNeur)->GetParent()->GetID();
				iDstNeurID 	= pCurEdge->GetDestinationID(pCurNeur);
				fEdgeValue 	= pCurEdge->GetValue();
				BZ2_bzWrite( &iBZ2Error, bz2out, &iDstLayerID, sizeof(int) );
				BZ2_bzWrite( &iBZ2Error, bz2out, &iDstNeurID, sizeof(int) );
				BZ2_bzWrite( &iBZ2Error, bz2out, &fEdgeValue, sizeof(float) );
			}
		}
		/*
		 * Only for back propagation networks
		 */
		if(bHasBias && fNetType == ANNetBP) {
			AbsNeuron *pCurNeur = pCurLayer->GetBiasNeuron();
			iNmbOfConnects = pCurNeur->GetConsO().size();
			BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfConnects, sizeof(int) );
			for(unsigned int k = 0; k < iNmbOfConnects; k++) {
				Edge *pCurEdge = pCurNeur->GetConO(k);
				iDstLayerID = pCurEdge->GetDestination(pCurNeur)->GetParent()->GetID();
				iDstNeurID = pCurEdge->GetDestinationID(pCurNeur);
				fEdgeValue = pCurEdge->GetValue();
				BZ2_bzWrite( &iBZ2Error, bz2out, &iDstLayerID, sizeof(int) );
				BZ2_bzWrite( &iBZ2Error, bz2out, &iDstNeurID, sizeof(int) );
				BZ2_bzWrite( &iBZ2Error, bz2out, &fEdgeValue, sizeof(float) );
			}
		}
	}

	BZ2_bzWriteClose ( &iBZ2Error, bz2out, 0, NULL, NULL );
	fclose( fout );
}

/*
 * GPGPU
 */
/*
F2DArray *Exporter::ExpToNArray() {
	BPLayer 	*pCurLayer 	= NULL;
	AbsNeuron	*pCurNeuron = NULL;

	if(	m_pNeurons ) {
		delete [] m_pNeurons;
		m_pNeurons = NULL;
	}
	F2DArray *pArray = new F2DArray;

	// Nr. of layers
	int iY 		= m_pNet->GetLayers().size();	// Nr. layers
	int iX 		= 0;							// Nr. Neurons
	int iCurX 	= 0;

	// 1. Check layer for the amount of neurons
	// 2. Look for the greatest layer
	// 3. Allocate memory based on the biggest layer
	for(int y = 0; y < iY; y++) {
		pCurLayer = ( (BPLayer*)m_pNet->GetLayer(y) );
		iCurX = pCurLayer->GetNeurons().size();
		if( iCurX > iX )
			iX = iCurX;
	}
	pArray->Alloc(iX, iY);
	std::cout 	<< "Neurons:\t " 	<< iX << std::endl
				<< "Layers:\t\t " 	<< iY << std::endl;

	// Save the values of each neuron into allocated memory
	for(int y = 0; y < iY; y++) {
		pCurLayer = ( (BPLayer*)m_pNet->GetLayer(y) );
		iCurX = pCurLayer->GetNeurons().size();
		// Neurons
		for(int x = 0; x < iCurX; x++) {
			pCurNeuron = pCurLayer->GetNeuron(x);
			pArray->SetValue(pCurNeuron->GetValue(), x, y);
		}
	}
	std::cout 	<< "2D array filled" << std::endl;
	m_pNeurons = pArray;
	return pArray;
}

F2DArray *Exporter::ExpToDArray() {
	BPLayer 	*pCurLayer 	= NULL;
	AbsNeuron	*pCurNeuron = NULL;

	if(	m_pDeltas ) {
		delete [] m_pDeltas;
		m_pDeltas = NULL;
	}
	F2DArray 	*pArray = new F2DArray;

	// nr. of layers
	int iY 		= m_pNet->GetLayers().size();	// Nr. layers
	int iX 		= 0;					// Nr. Neurons
	int iCurX 	= 0;

	// 1. Check layer for the amount of neurons
	// 2. Look for the greatest layer
	// 3. Allocate memory based on the biggest layer
	for(int y = 0; y < iY; y++) {
		pCurLayer = ( (BPLayer*)m_pNet->GetLayer(y) );
		iCurX = pCurLayer->GetNeurons().size();
		if( iCurX > iX )
			iX = iCurX;
	}
	pArray->Alloc(iX, iY);
	std::cout 	<< "Neurons:\t " 	<< iX << std::endl
				<< "Layers:\t\t " << iY << std::endl;

	// Save the values of each neuron into allocated memory
	for(int y = 0; y < iY; y++) {
		pCurLayer = ( (BPLayer*)m_pNet->GetLayer(y) );
		iCurX = pCurLayer->GetNeurons().size();
		// Neurons
		for(int x = 0; x < iCurX; x++) {
			pCurNeuron = pCurLayer->GetNeuron(x);
			pArray->SetValue(pCurNeuron->GetErrorDelta(), x, y);
		}
	}
	std::cout 	<< "2D array filled" << std::endl;
	m_pDeltas = pArray;
	return pArray;
}

F3DArray *Exporter::ExpToWArray() {
	BPLayer 	*pCurLayer 	= NULL;
	AbsNeuron	*pCurNeuron = NULL;

	if(	m_pWeights ) {
		delete [] m_pWeights;
		m_pWeights = NULL;
	}
	F3DArray 	*pArray = new F3DArray;

	// nr. of layers
	int iX 		= 0;							// Nr. Neurons
	int iY 		= m_pNet->GetLayers().size()-1;	// Nr. Layers
	int iZ 		= 0;							// Nr. Edges per Neuron

	int iCurX 	= 0;
	int iCurZ 	= 0;

	float fCurEdgeVal = 0.f;

	// Get dimension of the network
	// Layers
	for(int y = 0; y < iY; y++) {
		pCurLayer = ( (BPLayer*)m_pNet->GetLayer(y) );
		iCurX = pCurLayer->GetNeurons().size();
		if( iCurX > iX )
			iX = iCurX;
		// Neurons
		for(int x = 0; x < iCurX; x++) {
			pCurNeuron = pCurLayer->GetNeuron(x);
			iCurZ = pCurNeuron->GetConsO().size();
			if( iCurZ > iZ )
				iZ = iCurZ;
		}
	}
	pArray->Alloc(iX, iY, iZ);

	// Output
	std::cout 	<< "Neurons:\t " 	<< iX << std::endl
				<< "Layers:\t\t " << iY << std::endl
				<< "Edges p. Neuron: " 	<< iZ << std::endl;

	// FILL ARRAY (or Cube:)
	// Layers
	for(int y = 0; y < iY; y++) {
		pCurLayer = ( (BPLayer*)m_pNet->GetLayer(y) );
		iCurX = pCurLayer->GetNeurons().size();
		// Neurons
		for(int x = 0; x < iCurX; x++) {
			pCurNeuron = pCurLayer->GetNeuron(x);
			iCurZ = pCurNeuron->GetConsO().size();
			// Edges
			for(int z = 0; z < iCurZ; z++) {
				fCurEdgeVal = pCurNeuron->GetConO(z)->GetValue();
				pArray->SetValue(fCurEdgeVal, x, y, z);
			}
		}
	}

	// Output
	std::cout 	<< "3d array filled" << std::endl;
	m_pWeights = pArray;
	return pArray;
}
*/
