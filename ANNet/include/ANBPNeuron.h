/*
 * ANBPNeuron.h
 *
 *  Created on: 29.05.2009
 *      Author: dgrat
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <vector>

//own classes
#include <basic/ANAbsNeuron.h>

namespace ANN {

class Edge;
class AbsLayer;


/**
 * \brief Derived from ANAbsNeuron. Represents a neuron in a network.
 *
 * Pure virtual functions from abstract base are already implemented here.
 * You can modify the behavior of the complete net by overloading them.
 *
 * @author Daniel "dgrat" Frenzel
 */
class BPNeuron : public AbsNeuron {
private:
	float m_fLearningRate;	// 0,0 - 0,5
	float m_fWeightDecay;	// 0,005 - 0,03
	float m_fMomentum;		// 0,5 - 0,9

public:
	/*
	 * CTOR
	 * & DTOR
	 */
	BPNeuron(AbsLayer *parentLayer = NULL);
	/**
	 * Copy constructor for creation of a new neuron with the "same" properties like *pNeuron
	 * this constructor can't copy connections (edges), because they normally have dependencies to other neurons.
	 * @param pNeuron object to copy properties from
	 */
	BPNeuron(BPNeuron *pNeuron);
	~BPNeuron();

	/**
	 * Sets the scalar of the learning rate.
	 */
	void SetLearningRate 	(const float &fVal);
	/**
	 * Sets the scalar of the weight decay.
	 */
	void SetWeightDecay 	(const float &fVal);
	/**
	 * Sets the scalar of the momentum.
	 */
	void SetMomentum 		(const float &fVal);

	/**
	 * Defines how to calculate the values of each neuron.
	 */
	virtual void CalcValue();
	/**
	 * Defines how to calculate the error deltas of each neuron.
	 * Defines also how to change the weights.
	 */
	virtual void AdaptEdges();
};

}
#endif /* NEURON_H_ */
