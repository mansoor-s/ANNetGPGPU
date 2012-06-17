#ifndef NODE_H
#define NODE_H

#include <Qt/QtGui>

class Edge;
class Viewer;
class Layer;


class Node : public QGraphicsItem
{
private:
    int iWidth;

    bool m_bSelectedAsGroup;

    QList<Edge *> m_EdgeList;
    Viewer *m_pGraph;
    Layer *m_pLayer;

public:
    Node(Viewer *parent = NULL);
    virtual ~Node();

    void addEdge(Edge *edge);
    QList<Edge *> edges() const;

    void setLayer(Layer *layer);
    Layer* getLayer() const;

    float getWidth();

    void setSelectedAsGroup(bool b);
    bool selectedAsGroup();

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QRectF boundingRect() const;
    QPainterPath shape() const;

    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
};

#endif // NODE_H
