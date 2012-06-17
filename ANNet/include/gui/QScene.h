#ifndef SCENE_H
#define SCENE_H

#include <Qt/QtGui>

class Edge;
class Node;
class Layer;


class Scene : public QGraphicsScene
{
private:
    QList<Node*> m_lNodes;
    QList<Edge*> m_lEdges;
    QList<Layer*> m_lLayers;

public:
    Scene(QObject *parent = 0);

    void addEdge(Edge*);
    void removeEdge(Edge*);
    QList<Edge*> edges();

    void addNode(Node*);
    void removeNode(Node*);
    QList<Node*> nodes();

    Layer* addLayer(const unsigned int &iNodes, const QString &sName = "");
    void removeLayer(Layer*);
    QList<Layer*> layers();

    void adjust();
};

#endif // SCENE_H
