#ifndef LAYER_H
#define LAYER_H

#include <Qt/QtGui>

class Node;
class Scene;
class Viewer;
class Edge;
class Label;


class Layer : public QGraphicsItem
{
private:
    Viewer *m_pGraph;
    QList<Node *> m_NodeList;
    QRectF m_BoundingRect;
    QRectF m_LabelRect;
    Label *m_pLabel;
    Scene *m_pScene;

public:
    Layer(Viewer *parent = NULL);

    void addNode(Node *node);
    void addNodes(int iNeur);
    void removeNode(Node* pDelNode);
    QList<Node *> &nodes();

    void adjust();

    void shift(int dX, int dY);
    //void addNodes(const unsigned int &iNodes, const QString &sName);

    void setScene(Scene*);

    QList<Edge*> Connect(Layer*);
    QRectF getLabelRect();
    void setLabel(Label *pLabel);
    Label* getLabel();

    Label *addLabel(QString sName);

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QRectF boundingRect() const;

    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * event );
};

#endif // LAYER_H
