/*
 * SOMLayer.h
 *
 *  Created on: 26.02.2012
 *      Author: dgrat
 */

#ifndef SOMLAYER_H_
#define SOMLAYER_H_

#include <containers/AN2DArray.h>
#include <basic/ANAbsLayer.h>
#include <vector>

namespace ANN {

class SOMLayer : public AbsLayer {
private:
	std::vector<unsigned int> m_vDim;
	/*
	 * Flag describing the kind of layer.
	 * (i. e. input, hidden or output possible)
	 */
	LayerTypeFlag m_fTypeFlag;

public:
	SOMLayer();
	SOMLayer(const SOMLayer *pLayer);
	SOMLayer(const unsigned int &iSize, LayerTypeFlag fType);
	SOMLayer(const unsigned int &iWidth, const unsigned int &iHeight, LayerTypeFlag fType);
	SOMLayer(const std::vector<unsigned int> &vDim, LayerTypeFlag fType);
	virtual ~SOMLayer();

	virtual void Resize(const unsigned int &iSize);
	virtual void Resize(const unsigned int &iWidth, const unsigned int &iHeight);
	virtual void Resize(const std::vector<unsigned int> &vDim);

	/**
	 * Connects this layer with another one.
	 * Each neuron of this layer with each of the destination layer.
	 * @param pDestLayer pointer to layer to connect with.
	 * @param bAllowAdapt allows the change of the weights between both layers.
	 */
	void ConnectLayer(AbsLayer *pDestLayer, const bool &bAllowAdapt = true);

	/*
	 *
	 */
	void ConnectLayer(AbsLayer *pDestLayer, const F2DArray &f2dEdgeMat, const bool &bAllowAdapt = true);
	/**
	 * Sets learning rate scalar of the network.
	 * @param fVal New value of the learning rate. Recommended: 0.005f - 1.0f
	 */
	void SetLearningRate 	(const float &fVal);

	/**
	 *
	 */
	std::vector<unsigned int> GetDim() const;
	unsigned int GetDim(const unsigned int &iInd) const;
};

}

#endif /* SOMLAYER_H_ */
