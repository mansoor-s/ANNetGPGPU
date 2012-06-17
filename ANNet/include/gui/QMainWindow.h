#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <Qt/QtGui>
#include <gui/QViewer.h>
#include <gui/QScene.h>
#include <gui/fancytabwidget.h>
#include <gui/QInputdialog.h>
#include <gui/QCustomPlot/qcustomplot.h>

using namespace Core;
using namespace Core::Internal;


class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    FancyTabWidget *m_pTabBar;
    QToolBar *m_ActionsBar;

    QAction *m_pAddLayer;
    QAction *m_pAddNeuron;
    QAction *m_pAddEdges;

    QAction *m_pRemoveLayers;
    QAction *m_pRemoveNeurons;

    QAction *m_pRemoveEdges;
    QAction *m_pRemoveAllEdges;

    Viewer *m_pViewer;
    QCustomPlot *m_pCustomPlot;
    InputDialog *m_pInputDial;

    QMenu *m_pFileMenu;
    QAction *m_pSave;
    QAction *m_pLoad;
    QAction *m_pNew;
    
public slots:
    void sl_createLayer();

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void createMenus();
    void createTabs();
    void createActions();
};

#endif // MAINWINDOW_H
