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
