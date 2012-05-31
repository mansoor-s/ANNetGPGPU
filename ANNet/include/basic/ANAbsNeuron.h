/*
 * ANAbsNeuron.h
 *
 *  Created on: 01.09.2010
 *      Author: dgrat
 */

#ifndef ABSNEURON_H_
#define ABSNEURON_H_

#include <vector>
#include <string>

#include <basic/ANExporter.h>
#include <basic/ANImporter.h>

namespace ANN {

// containers
class F2DArray;
class F3DArray;
class TrainingSet;
class ConTable;
// math
class TransfFunction;
// net
class AbsLayer;
class AbsNeuron;
class Edge;


/**
 * \brief Abstract class describing a basic neuron in a network.
 *
 * Pure virtual functions must get implemented if deriving from this class.
 * These functions a doing the back-/propagation jobs.
 * You can modify the behavior of the complete net by overloading them.
 *
 * @author Daniel "dgrat" Frenzel
 */
class AbsNeuron {
protected:
	std::vector<float> 	m_vPosition;		// x, y, z, .. coordinates of the neuron (e.g. SOM)
	float m_fValue;							// value of the neuron in the net

	float m_fErrorDelta;					// Current error delta of this neuron

	AbsLayer *m_pParentLayer;				// layer which is inheriting this neuron
	int m_iNeuronID;						// ID of this neuron in the layer

	Edge *m_pBias;							// Pointer to the bias edge (or connection to bias neuron)

	std::vector<Edge*> m_lOutgoingConnections;
	std::vector<Edge*> m_lIncomingConnections;

	const TransfFunction *m_ActFunction;

public:
	/**
	 * Creates a new neuron with parent layer: *pParentLayer
	 */
	AbsNeuron(AbsLayer *pParentLayer = NULL);
	/**
	 * Copy constructor for creation of a new neuron with the "same" properties like *pNeuron
	 * this constructor can't copy connections (edges), because they normally have dependencies to other neurons.
	 * @param pNeuron object to copy properties from
	 */
	AbsNeuron(const AbsNeuron *pNeuron);
	virtual ~AbsNeuron();

	/*
	 * TODO
	 */
	void EraseAllEdges();

	/**
	 * Pointer to the layer inherting this neuron.
	 */
	AbsLayer *GetParent() const;

	/**
	 * Appends an edge to the list of incoming edges.
	 */
	virtual void AddConI(Edge *ANEdge);
	/**
	 * Appends an edge to the list of outgoing edges.
	 */
	virtual void AddConO(Edge *ANEdge);

	void SetConO(Edge *Edge, const unsigned int iID);
	void SetConI(Edge *Edge, const unsigned int iID);

	/**
	 * @return Pointer to an incoming edge
	 * @param iID Index of edge in m_lIncomingConnections
	 */
	Edge* GetConI(const unsigned int &iID) const;
	/**
	 * @return Pointer to an outgoing edge
	 * @param iID Index of edge in m_lOutgoingConnections
	 */
	Edge* GetConO(const unsigned int &iID) const;
	/**
	 * @return Array of pointers of all incoming edges
	 */
	std::vector<Edge*> GetConsI() const;
	/**
	 * @return Array of pointers of all outgoing edges
	 */
	std::vector<Edge*> GetConsO() const;
	/**
	 * @param iID New index of this neuron.
	 */
	virtual void SetID(const int iID);
	/**
	 * @return Index of this neuron.
	 */
	unsigned int GetID() const;
	/**
	 * @param fValue New value of this neuron.
	 */
	virtual void SetValue(const float &fValue);
	/**
	 * @return Returns the value of this neuron.
	 */
	const float &GetValue() const;
	/*
	 * Get the position of the neuron
	 * @return x, y, z, .. coordinates of the neuron (e.g. SOM)
	 */
	const std::vector<float> GetPosition() const;
	/**
	 * Sets the current position of the neuron in the net.
	 * @param vPos Vector with Cartesian coordinates
	 */
	void SetPosition(const std::vector<float> &vPos);
	/**
	 * @param fValue New error delts of this neuron.
	 */
	virtual void SetErrorDelta(const float &fValue);
	/**
	 * @return Returns the error delta of this neuron.
	 */
	const float &GetErrorDelta() const;
	/**
	 * @param pANEdge Pointer to edge connecting this neuron with bias neuron.
	 */
	virtual void SetBiasEdge(Edge *pANEdge);
	/**
	 * @return Returns pointer to edge connecting this neuron with bias neuron.
	 */
	Edge *GetBiasEdge() const;
	/**
	 * @param pFCN Kind of function the net has to use while back-/propagating.
	 */
	virtual void SetTransfFunction (const TransfFunction *pFCN);
	/**
	 * @return Return the kind of function the net has to use while back-/propagating.
	 */
	const TransfFunction *GetTransfFunction() const;

	/**
	 * Overload to define how the net has to act while propagating back.
	 * I. e. how to modify the edges after calculating the error deltas.
	 */
	virtual void AdaptEdges() 	= 0;
	/**
	 * Overload to define how the net has to act while propagating.
	 * I. e. which neurons/edges to use for calculating the new value of the neuron
	 */
	virtual void CalcValue() 		= 0;

	/* QUASI STATIC:*/

	/**
	 * standard output of the net. Only usable if input/output layer was already set.
	 */
	friend std::ostream& operator << (std::ostream &os, AbsNeuron &op);
	/**
	 * standard output of the net. Only usable if input/output layer was already set.
	 */
	friend std::ostream& operator << (std::ostream &os, AbsNeuron *op);

	/**
	 * Connects a neuron with another neuron
	 * bAdaptState indicates whether the connection is changeable
	 */
	friend void Connect(AbsNeuron *pSrcNeuron, AbsNeuron *pDstNeuron, const bool &bAdaptState);
	/**
	 * Connects a neuron with a complete layer
	 * bAdaptState indicates whether the connections are changeable
	 */
	friend void Connect(AbsNeuron *pSrcNeuron, AbsLayer  *pDestLayer, const bool &bAdaptState);
	/**
	 * Connects a neuron with another neuron
	 * Allows to set the value of the connection and the current momentum
	 * bAdaptState indicates whether the connection is changeable
	 */
	friend void Connect(AbsNeuron *pSrcNeuron, AbsNeuron *pDstNeuron, const float &fVal, const float &fMomentum, const bool &bAdaptState);
	/**
	 *
	 */
	friend void Connect(AbsNeuron *srcNeuron, AbsLayer *destLayer, const std::vector<float> &vValues, const std::vector<float> &vMomentums, const bool &bAdaptState);

	operator float() const;
};

}
#endif /* ABSNEURON_H_ */
