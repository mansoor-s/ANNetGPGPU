/*
 * ANConTable.h
 *
 *  Created on: 22.01.2011
 *      Author: dgrat
 */

#ifndef NETCONNECTIONTABLE_H_
#define NETCONNECTIONTABLE_H_

#include <stdint.h>

namespace ANN {

typedef uint32_t LayerTypeFlag;
typedef uint32_t NetTypeFlag;

/**
 * \brief Represents a container for a connection (edge/weight) in the network.
 *
 * @author Daniel "dgrat" Frenzel
 */
struct ConDescr {
	int m_iSrcLayerID;
	int m_iDstLayerID;

	int m_iSrcNeurID;
	int m_iDstNeurID;

	float m_fVal;
};

struct NeurDescr {
	int m_iLayerID;
	int m_iNeurID;
	std::vector<float> m_vPos;
};

/**
 * \brief Represents a container for all connections (edges/weights) in the network.
 *
 * @author Daniel "dgrat" Frenzel
 */
struct ConTable {
	NetTypeFlag 				NetType;
	unsigned int 				NrOfLayers;

	std::vector<unsigned int> 	SizeOfLayer;
	std::vector<LayerTypeFlag> 	TypeOfLayer;

	std::vector<NeurDescr> 		Neurons;

	std::vector<ConDescr> 		BiasCons;		// TODO not elegant
	std::vector<ConDescr> 		NeurCons;
};

}
#endif /* NETCONNECTIONTABLE_H_ */
