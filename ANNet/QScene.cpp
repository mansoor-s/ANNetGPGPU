#include <gui/QScene.h>
#include <gui/QNode.h>
#include <gui/QEdge.h>
#include <gui/QLayer.h>
#include <gui/QLabel.h>
#include <iostream>


Scene::Scene(QObject *parent) : QGraphicsScene(parent)
{
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
    m_lLayers << pLayer;
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
