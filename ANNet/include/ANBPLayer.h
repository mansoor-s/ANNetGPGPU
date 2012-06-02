/*
 * ANBPLayer.h
 *
 *  Created on: 02.06.2009
 *      Author: dgrat
 */

#ifndef SIMPLELAYER_H_
#define SIMPLELAYER_H_

#include <iostream>
#include <vector>
#include <stdint.h>

#include <basic/ANAbsLayer.h>

namespace ANN {

// own classes
class Function;
class BPNeuron;
class ConTable;


/**
 * \brief Represents a container for neurons in a back propagation network.
 *
 * Functions implemented here allow us to connect neurons with each other.
 *
 * @author Daniel "dgrat" Frenzel
 */
class BPLayer : public AbsLayer {
	/*
	 * Pointer to bias neuron.
	 */
	BPNeuron *m_pBiasNeuron;

public:
	/**
	 * Creates a new layer
	 */
	BPLayer();
	/**
	 * Copy constructor for creation of a new layer with the "same" properties like *pLayer
	 * this constructor can't copy connections (edges), because they normally have dependencies to other layers.
	 * @param pLayer object to copy properties from
	 */
	BPLayer(const BPLayer *pLayer);
	/**
	 * Creates a new layer
	 * @param iNumber Number of neurons of this layer.
	 *
	 * @param fType Flag describing the type of the layer.
	 */
	BPLayer(const unsigned int &iNumber, LayerTypeFlag fType);
	virtual ~BPLayer();

	/**
	 * Resizes the layer. Deletes old neurons and adds new ones (initialized with random values).
	 * @param iSize New number of neurons.
	 * @param iShiftID Value which has to get added to the ID of each neuron.
	 */
	virtual void Resize(const unsigned int &iSize);

	/**
	 * Sets the type of the layer (input, hidden or output layer)
	 * @param fType Flag describing the type of the layer.
	 * Flag: "ANBiasNeuron" will automatically add a bias neuron.
	 */
	virtual void SetFlag(const LayerTypeFlag &fType);
	/**
	 * Adds a flag if not already set.
	 * @param fType Flag describing the type of the layer.
	 * Flag: "ANBiasNeuron" will automatically add a bias neuron.
	 */
	virtual void AddFlag(const LayerTypeFlag &fType);

	/**
	 * Pointer to the Bias neuron.
	 * @return Return the pointer of the bias neuron in this layer
	 */
	BPNeuron *GetBiasNeuron() const;

	/**
	 * Connects this layer with another one "pDestLayer".
	 * @param pDestLayer is a pointer of the layer to connect with.
	 * @param Connections is an array which describes these connections.
	 * The first index of this array is equal to the ID of the neuron in the actual (this) layer.
	 * The second ID is equal to the ID of neurons in the other (pDestLayer).
	 * @param bAllowAdapt allows the change of the weights between both layers.
	 */
	void ConnectLayer(AbsLayer *pDestLayer, std::vector<std::vector<int> > Connections, const bool bAllowAdapt = true); // TODO use connections table
	/**
	 * Connects this layer with another one.
	 * Each neuron of this layer with each of the neurons in "pDestLayer".
	 * Neurons in "pDestLayer" get
	 * @param pDestLayer pointer to layer to connect with.
	 * @param bAllowAdapt allows the change of the weights between both layers.
	 */
	void ConnectLayer(AbsLayer *pDestLayer, const bool &bAllowAdapt = true);

	/**
	 * Sets learning rate scalar of the network.
	 * @param fVal New value of the learning rate. Recommended: 0.005f - 1.0f
	 */
	void SetLearningRate 	(const float &fVal);
	/**
	 * Sets momentum scalar of the network.
	 * @param fVal New value of the momentum. Recommended: 0.3f - 0.9f
	 */
	void SetMomentum 		(const float &fVal);
	/**
	 * Sets weight decay of the network.
	 * @param fVal New value of the weight decay. Recommended: 0.f
	 */
	void SetWeightDecay 	(const float &fVal);

	/**
	 * Save layer's content to filesystem
	 */
	virtual void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	/**
	 * Load layer's content to filesystem
	 * @return The ID of the current layer.
	 */
	virtual int ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table);

	/**
	 * standard output of the layer.
	 */
	friend std::ostream& operator << (std::ostream &os, BPLayer &op);
	/**
	 * standard output of the layer.
	 */
	friend std::ostream& operator << (std::ostream &os, BPLayer *op);
};

}

#endif /* SIMPLELAYER_H_ */
