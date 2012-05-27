/*
 * ANHFNeuron.h
 *
 *  Created on: 22.02.2011
 *      Author: dgrat
 */

#ifndef ANHFNEURON_H_
#define ANHFNEURON_H_

#include <basic/ANAbsNeuron.h>

namespace ANN {


class HFNeuron : public AbsNeuron {
public:
	HFNeuron(AbsLayer *parentLayer = NULL);
	virtual ~HFNeuron();

	/**
	 * Defines how to calculate the values of each neuron.
	 */
	virtual void CalcValue();

	/**
	 * Unused function in this hopfield net.
	 */
	virtual void AdaptEdges();
};

}

#endif /* ANHFNEURON_H_ */
