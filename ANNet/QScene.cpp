#include <gui/QScene.h>
#include <gui/QNode.h>
#include <gui/QEdge.h>
#include <gui/QLayer.h>
#include <gui/QLabel.h>
#include <gui/QZLabel.h>
#include <containers/ANConTable.h>
#include <math/ANRandom.h>
#include <iostream>
#include <cassert>


Scene::Scene(QObject *parent) : QGraphicsScene(parent)
{
	m_pANNet = new ANN::BPNet;
}

ANN::BPNet *Scene::getANNet() {
	m_pANNet->EraseAll();

	int LayerTypeFlag 	= -1;
	int iSize 			= -1;

	/**
	 * Create layers for neural net
	 */
	QList<ANN::BPLayer*> lLayers;
	foreach(Layer *pLayer, m_lLayers) {
		LayerTypeFlag = pLayer->getLabel()->getType();
		iSize = pLayer->nodes().size();

		assert(iSize > 0);	// shouldn't happen

		int iZ = pLayer->getZLabel()->getZLayer();
		/*
		if(iZ < 0) {
			QMessageBox msgBox;
			msgBox.setText("Z-values must be set for all layers.");
			msgBox.exec();

			return NULL;
		}

		if(LayerTypeFlag < 0) {
			QMessageBox msgBox;
			msgBox.setText("Type of layer must be set for all layers.");
			msgBox.exec();

			return NULL;
		}*/
/*
		ANN::BPLayer *pBPLayer = new ANN::BPLayer(iSize, LayerTypeFlag);
		pBPLayer->SetZLayer(iZ);
		lLayers << pBPLayer;
*/
	}
	
   /**
	* Build connections
	*/
	ANN::ConTable Net;
	Net.NetType 	= ANN::ANNetBP;
	Net.NrOfLayers 	= m_lLayers.size();

	//std::cout<<"number of layers: "<<m_lLayers.size()<<std::endl;
	foreach(Layer *pLayer, m_lLayers) {
		Net.SizeOfLayer.push_back(pLayer->nodes().size() );
		Net.ZValOfLayer.push_back(pLayer->getZLabel()->getZLayer() );
		Net.TypeOfLayer.push_back(pLayer->getLabel()->getType() );

		std::cout<<"zlayer: "<<pLayer->getZLabel()->getZLayer()<<std::endl;

		//std::cout<<"number of neurons: "<<pLayer->nodes().size()<<std::endl;
		foreach(Node *pNode, pLayer->nodes() ) {
			ANN::NeurDescr neuron;
			neuron.m_iLayerID = pLayer->getID();
			neuron.m_iNeurID = pNode->getID();
			Net.Neurons.push_back(neuron);

			//std::cout<<"number of edges O: "<<pNode->edgesO().size()<<std::endl;
			foreach(Edge *pEdge, pNode->edgesO() ) {
				ANN::ConDescr edge;
				edge.m_iSrcLayerID 	= pEdge->sourceNode()->getLayer()->getID();
				edge.m_iDstLayerID 	= pEdge->destNode()->getLayer()->getID();
				edge.m_iSrcNeurID 	= pEdge->sourceNode()->getID();
				edge.m_iDstNeurID 	= pEdge->destNode()->getID();
				edge.m_fVal 		= ANN::RandFloat(-0.5f, 0.5f);

				Net.NeurCons.push_back(edge);
			}
		}
	}
	m_pANNet->CreateNet(Net);

	/**
	 * Add layers to neural net
	 */
	foreach(ANN::BPLayer *pLayer, lLayers) {
		m_pANNet->AddLayer(pLayer);
	}

	return m_pANNet;
}

void Scene::setANNet(ANN::BPNet &) {

}

void Scene::adjust() {
    foreach (Edge *edge, m_lEdges)
        edge->adjust();
    foreach (Layer *layer, m_lLayers)
        layer->adjust();
}

Layer* Scene::addLayer(const unsigned int &iNodes, const QString &sName) {
    Layer *pLayer = new Layer;
    pLayer->setScene(this);

    for(unsigned int i = 0; i < iNodes; i++) {
        Node *pNode = new Node;
        pNode->setPos(i*(pNode->getWidth()+8), 0);
        pLayer->addNode(pNode);

        pNode->setLayer(pLayer);
        addItem(pNode);
    }
    addItem(pLayer);
    addItem(pLayer->addLabel(sName));
    addItem(pLayer->addZLabel(-1));
    m_lLayers << pLayer;
    pLayer->setID(m_lLayers.size()-1);
    pLayer->adjust();

    return pLayer;
}

void Scene::addNode(Node* pNode) {
    m_lNodes << pNode;
    addItem(pNode);
}

void Scene::addEdge(Edge* pEdge) {
    m_lEdges << pEdge;
    addItem(pEdge);
}

void Scene::removeEdge(Edge* pDelEdge) {
    removeItem(pDelEdge);

    QList<Edge*> pNewList;
    foreach(Edge *pEdge, m_lEdges) {
        if(pEdge != pDelEdge)
            pNewList << pEdge;
    }
    m_lEdges = pNewList;
}

void Scene::removeNode(Node* pDelNode) {
    removeItem(pDelNode);
    pDelNode->getLayer()->removeNode(pDelNode);

    QList<Node*> pNewList;
    foreach(Node *pNode, m_lNodes) {
        if(pNode != pDelNode)
            pNewList << pNode;
    }
    m_lNodes = pNewList;
}

void Scene::removeLayer(Layer* pDelLayer) {
	removeItem(pDelLayer->getZLabel());
    removeItem(pDelLayer->getLabel());
    removeItem(pDelLayer);

    foreach(Node *pNode, pDelLayer->nodes()) {
        removeNode(pNode);
    }

    QList<Layer*> pNewList;
    foreach(Layer *pLayer, m_lLayers) {
        if(pLayer != pDelLayer)
            pNewList << pLayer;
    }
    m_lLayers = pNewList;
}

QList<Edge*> Scene::edges() {
    return m_lEdges;
}

QList<Node*> Scene::nodes() {
    return m_lNodes;
}

QList<Layer*> Scene::layers() {
    return m_lLayers;
}
