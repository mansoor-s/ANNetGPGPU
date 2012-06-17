#ifndef LABEL_H
#define LABEL_H

#include <Qt/QtGui>


class Label : public QGraphicsItem
{
private:
    QRectF m_BRect;
    QString m_sName;

public:
    Label();
    void setBRect(QRectF rect);
    void SetName(QString sName);
    QString GetName();

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QRectF boundingRect() const;

//    virtual void mousePressEvent ( QGraphicsSceneMouseEvent *event );
    virtual void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * event );
};

#endif // LABEL_H
