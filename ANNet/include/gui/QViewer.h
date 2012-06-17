#ifndef VIEWER_H
#define VIEWER_H

#include <Qt/QtGui>
#include <ANNet>

class Scene;
class Node;


class Viewer : public QGraphicsView //, public QObject
{
    Q_OBJECT

private:
    Scene *m_pScene;
    QList<Node*> m_lNodesSrc;
    QList<Node*> m_lNodesDest;

    bool m_bStartConnect;

private slots:
    void sl_waitForDest();

signals:
    void si_selectionChanged();

public:
    Viewer(QWidget *parent = 0);
    virtual ~Viewer();

    Scene *getScene();

    ANN::BPNet getANNet();

public slots:
    void sl_addNeurons();
    void sl_createConnections();

    void sl_removeLayers();
    void sl_removeNeurons();

    void sl_removeConnections();
    void sl_removeAllConnections();

protected:
    virtual void mouseMoveEvent( QMouseEvent * event );
    virtual void mousePressEvent( QMouseEvent * event );
    virtual void mouseReleaseEvent( QMouseEvent * event );

    virtual void keyPressEvent ( QKeyEvent * event );
    virtual void wheelEvent(QWheelEvent* event);

    //Holds the current centerpoint for the view, used for panning and zooming
    QPointF CurrentCenterPoint;

    //From panning the view
    QPoint LastPanPoint;

    //Set the current centerpoint in the
    void SetCenter(const QPointF& centerPoint);
    QPointF GetCenter() { return CurrentCenterPoint; }
};

#endif // VIEWER_H
