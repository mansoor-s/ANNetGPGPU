/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <Qt/QtGui>
// own classes
#include <ANNet>
#include <containers/ANTrainingSet.h>
#include <gui/QViewer.h>
#include <gui/QScene.h>
#include <gui/QTrainingForm.h>
#include <gui/QIOForm.h>
#include <gui/QOutput.h>

//3rd party classes
#include <gui/QCustomPlot/qcustomplot.h>
#include <gui/fancytabwidget.h>
#include <gui/fancyactionbar.h>

using namespace Core;
using namespace Core::Internal;


class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    ANN::BPNet *m_pANNet;
    ANN::TrainingSet m_TrainingSet;

    /////////////////////////////////////////
    FancyActionBar *m_pActionBar;

    QAction *m_pStartTraining;
    QAction *m_pRunInput;

    /////////////////////////////////////////
    QToolBar *m_ActionsBar;

    QAction *m_pAddLayer;
    QAction *m_pAddNeuron;
    QAction *m_pAddEdges;

    QAction *m_pRemoveLayers;
    QAction *m_pRemoveNeurons;

    QAction *m_pRemoveEdges;
    QAction *m_pRemoveAllEdges;

    /////////////////////////////////////////
    FancyTabWidget *m_pTabBar;

    Viewer *m_pViewer;
    QCustomPlot *m_pCustomPlot;
    IOForm *m_pInputDial;
    TrainingForm *m_pTrainingDial;
    Output *m_pOutputTable;

    /////////////////////////////////////////
    QMenu *m_pFileMenu;

    QAction *m_pSave;
    QAction *m_pLoad;
    QAction *m_pNew;
    QAction *m_pQuit;
    
    QMenu *m_pViewMenu;
    QAction *m_pZoomIn;
    QAction *m_pZoomOut;
    QAction *m_pShowEdges;
    QAction *m_pShowNodes;

    /////////////////////////////////////////
    std::vector<float> m_vErrors;

public slots:
    void sl_createLayer();
    void sl_startTraining();
    void sl_run();
    void sl_setTrainingSet();

    // File menu
    void sl_newProject();
    void sl_saveANNet();
    void sl_loadANNet();

    // View menu
    void sl_zoomIn();
    void sl_zoomOut();
    void sl_ShowEdges(bool);
    void sl_ShowNodes(bool);

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void createGraph();
    void createMenus();
    void createTabs();
    void createActions();
};

#endif // MAINWINDOW_H
