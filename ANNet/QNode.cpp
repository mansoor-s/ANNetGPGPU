#include <gui/QNode.h>
#include <gui/QEdge.h>
#include <gui/QViewer.h>
#include <gui/QLayer.h>


Node::Node(Viewer *parent)
{
    setFlag(QGraphicsItem::ItemIsMovable);
    setFlag(QGraphicsItem::ItemIsSelectable);

    m_bSelectedAsGroup = false;

    m_pGraph = parent;
    iWidth = 20;
    m_pLayer = NULL;

    setZValue(2);
}

Node::~Node()
{
}

void Node::setSelectedAsGroup(bool b) {
    m_bSelectedAsGroup = b;
}

bool Node::selectedAsGroup() {
    return m_bSelectedAsGroup;
}

float Node::getWidth() {
    return iWidth;
}

void Node::addEdge(Edge *edge)
{
    m_EdgeList << edge;
    edge->adjust();
}

QList<Edge *> Node::edges() const
{
    return m_EdgeList;
}

void Node::setLayer(Layer *layer) {
    m_pLayer = layer;
}

Layer* Node::getLayer() const {
    return m_pLayer;
}

void Node::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    QRadialGradient gradient(-3, -3, iWidth/2);
    if (option->state & QStyle::State_Sunken) {
        gradient.setCenter(3, 3);
        gradient.setFocalPoint(3, 3);
        gradient.setColorAt(1, QColor(Qt::yellow).light(iWidth));
        gradient.setColorAt(0, QColor(Qt::darkYellow).light(iWidth));
    }
    else if(option->state & QStyle::State_Selected) {
        gradient.setColorAt(0, Qt::white);
        gradient.setColorAt(1, Qt::red);

        foreach (Edge *edge, m_EdgeList) {
            edge->setColor(Qt::red);
            edge->setZValue(1);
        }
    }
    else if(m_bSelectedAsGroup) {
        gradient.setColorAt(0, Qt::white);
        QColor orange(255, 128, 0);
        gradient.setColorAt(1, orange);
    }
    else {
        gradient.setColorAt(0, Qt::yellow);
        gradient.setColorAt(1, Qt::darkYellow);

        foreach (Edge *edge, m_EdgeList) {
            edge->setColor(Qt::black);
            edge->setZValue(0);
        }
    }
    painter->setBrush(gradient);

    painter->setPen(QPen(Qt::black, 0));
    painter->drawEllipse(-iWidth/2, -iWidth/2, iWidth, iWidth);
}

QPainterPath Node::shape() const
{
    QPainterPath path;
    path.addEllipse(-iWidth/2, -iWidth/2, iWidth, iWidth);
    return path;
}

QRectF Node::boundingRect() const {
    qreal adjust = 2;
    return QRectF( -iWidth/2 - adjust, -iWidth/2 - adjust, iWidth+3 + adjust, iWidth+3 + adjust);
}

void Node::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    foreach (Edge *edge, m_EdgeList)
        edge->adjust();

    if(m_pLayer != NULL)
        m_pLayer->adjust();

    update();
    QGraphicsItem::mousePressEvent(event);
}

void Node::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    foreach (Edge *edge, m_EdgeList)
        edge->adjust();

    if(m_pLayer != NULL)
        m_pLayer->adjust();

    update();
    QGraphicsItem::mouseReleaseEvent(event);
}

void Node::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    foreach (Edge *edge, m_EdgeList)
        edge->adjust();

    if(m_pLayer != NULL)
        m_pLayer->adjust();

    update();
    QGraphicsItem::mouseMoveEvent(event);
}
