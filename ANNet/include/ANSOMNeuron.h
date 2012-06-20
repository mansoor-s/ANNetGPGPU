/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/

#ifndef SOMNEURON_H_
#define SOMNEURON_H_

#include <vector>
#include <basic/ANAbsNeuron.h>

namespace ANN {

class SOMLayer;


class SOMNeuron : public AbsNeuron {
protected:
	float 				m_fLearningRate;
	float 				m_fInfluence;

public:
	SOMNeuron(SOMLayer *parent = 0);
	virtual ~SOMNeuron();

	/**
	 * Overload to define how the net has to act while propagating back.
	 * I. e. how to modify the edges after calculating the error deltas.
	 */
	virtual void AdaptEdges();

	/**
	 * Calculates the value of the neuron
	 */
	virtual void CalcValue();

	/**
	 * Calculates the distance of the neuron to the input vector
	 */
	virtual void CalcDistance2Inp();

	/**
	 * @return Returns the current learning rate
	 */
	float GetLearningRate() const;

	/**
	 * Sets the learning rate
	 * @param fVal Current learning rate
	 */
	void SetLearningRate(const float &fVal);

	/**
	 * @return Returns the current influence
	 */
	float GetInfluence() const;

	/**
	 * Sets the learning rate
	 * @param fVal Current influence
	 */
	void SetInfluence(const float &fVal);

	/**
	 * @return Returns the current distance of the neuron to its input vector.
	 */
	float GetDistance2Neur(const SOMNeuron &pNeurDst);

	/**
	 * @return Returns the current distance of the neuron to its input vector.
	 */
	friend float GetDistance2Neur(const SOMNeuron &pNeurSrc, const SOMNeuron &pNeurDst);
};

}

#endif /* SOMNEURON_H_ */
