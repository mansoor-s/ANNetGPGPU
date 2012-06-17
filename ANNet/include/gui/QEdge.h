#ifndef EDGE_H
#define EDGE_H

#include <Qt/QtGui>

class Node;


class Edge : public QGraphicsItem
{
private:
    Node *m_pSource, *m_pDest;

    QPointF m_SourcePoint;
    QPointF m_DestPoint;
    qreal m_ArrowSize;

    QColor m_Color;

public:
    Edge(Node *pSourceNode, Node *pDestNode);
    virtual ~Edge();

    Node *sourceNode() const;
    Node *destNode() const;

    void adjust();

    void setColor(QColor color);

protected:
    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
};

#endif // EDGE_H
