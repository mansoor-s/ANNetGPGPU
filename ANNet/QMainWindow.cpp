#include <iostream>
#include <gui/QLayer.h>
#include <gui/QMainWindow.h>
#include <gui/utils/stylehelper.h>
#include <gui/utils/manhattanstyle.h>  //"manhattanstyle.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    using namespace Core;
    using namespace Core::Internal;

    QCoreApplication::setApplicationName(QLatin1String("ANNetDesigner"));
    QString baseName = QApplication::style()->objectName();
#ifdef Q_WS_X11
    if (baseName == QLatin1String("windows")) {
        // Sometimes we get the standard windows 95 style as a fallback
        // e.g. if we are running on a KDE4 desktop
        QByteArray desktopEnvironment = qgetenv("DESKTOP_SESSION");
        if (desktopEnvironment == "kde")
            baseName = QLatin1String("plastique");
        else
            baseName = QLatin1String("cleanlooks");
    }
#endif
    qApp->setStyle(new ManhattanStyle(baseName));
    Utils::StyleHelper::setBaseColor(Qt::darkGray);

    m_ActionsBar    = new QToolBar;
    m_pTabBar       = new FancyTabWidget;
    m_pActionBar 	= new FancyActionBar;

    m_pViewer       = new Viewer;
    m_pCustomPlot   = new QCustomPlot;
    m_pInputDial    = new IOForm;
    m_pTrainingDial = new TrainingForm;

    m_pNew          = new QAction(tr("New project"), 0);
    m_pSave         = new QAction(tr("Save project"), 0);
    m_pLoad         = new QAction(tr("Load project"), 0);

    setCentralWidget(m_pTabBar);
    addToolBar(m_ActionsBar);

    createTabs();
    createMenus();
    createActions();
}

MainWindow::~MainWindow()
{
    
}

void MainWindow::createTabs() {
    m_pTabBar->insertTab(0, m_pViewer, QIcon("gfx/monitor_icon.png"),"Network designer" );
    m_pTabBar->setTabEnabled(0, true);
    m_pTabBar->insertTab(1, m_pInputDial, QIcon("gfx/training_icon.png"),"Input/Output data" );
    m_pTabBar->setTabEnabled(1, true);
    m_pTabBar->insertTab(2, m_pTrainingDial, QIcon("gfx/QuestionMark.png"),"Training procedure" );
    m_pTabBar->setTabEnabled(2, true);
    m_pTabBar->insertTab(3, m_pCustomPlot, QIcon("gfx/graph_icon.png"),"Learning curve" );
    m_pTabBar->setTabEnabled(3, true);

    m_pTabBar->setCurrentIndex(0);
    m_pTabBar->addCornerWidget(m_pActionBar);
}

void MainWindow::createMenus() {
    m_pFileMenu = menuBar()->addMenu(tr("&File"));
    m_pFileMenu->addAction(m_pNew);
    m_pFileMenu->addAction(m_pSave);
    m_pFileMenu->addAction(m_pLoad);
}

void MainWindow::createActions() {
    QIcon iconLayer("gfx/layer.png");
    QIcon iconNeuron("gfx/neuron.png");
    QIcon iconEdge("gfx/edge.png");

    QIcon iconRemNeuron("gfx/rem_neuron.png");
    QIcon iconRemLayer("gfx/rem_layer.png");

    QIcon iconRemEdge("gfx/rem_edge.png");
    QIcon iconRemEdges("gfx/rem_edges.png");

    QIcon iconStartTraining("gfx/arrow.png");

    /*
     * Fancy action bar
     */
    m_pStartTraining = new QAction(iconStartTraining, "Start Training", 0);
    m_pActionBar->insertAction(0, m_pStartTraining);

    connect(m_pStartTraining, SIGNAL(triggered ()), this, SLOT(sl_startTraining()) );

    /*
     * Regular tool bar
     */
    m_pAddLayer = m_ActionsBar->addAction(iconLayer, "Add a layer");
    m_pRemoveLayers = m_ActionsBar->addAction(iconRemLayer, "Remove selected layers");
    m_ActionsBar->addSeparator();
    m_pAddNeuron = m_ActionsBar->addAction(iconNeuron, "Add neurons to selected layers");
    m_pRemoveNeurons = m_ActionsBar->addAction(iconRemNeuron, "Remove selected neurons");
    m_ActionsBar->addSeparator();
    m_pAddEdges = m_ActionsBar->addAction(iconEdge, "Add edges to selected neurons");
    m_pRemoveEdges = m_ActionsBar->addAction(iconRemEdge, "Remove selected edges");
    m_ActionsBar->addSeparator();
    m_pRemoveAllEdges = m_ActionsBar->addAction(iconRemEdges, "Remove all edges");

    connect(m_pAddLayer, SIGNAL(triggered ()), this, SLOT(sl_createLayer()) );
    connect(m_pAddNeuron, SIGNAL(triggered ()), m_pViewer, SLOT(sl_addNeurons()) );
    connect(m_pAddEdges, SIGNAL(triggered ()), m_pViewer, SLOT(sl_createConnections()) );

    connect(m_pRemoveLayers, SIGNAL(triggered ()), m_pViewer, SLOT(sl_removeLayers()) );
    connect(m_pRemoveNeurons, SIGNAL(triggered() ), m_pViewer, SLOT(sl_removeNeurons()) );

    connect(m_pRemoveEdges, SIGNAL(triggered ()), m_pViewer, SLOT(sl_removeConnections()) );
    connect(m_pRemoveAllEdges, SIGNAL(triggered() ), m_pViewer, SLOT(sl_removeAllConnections()) );
}

void MainWindow::sl_startTraining() {

}

void MainWindow::sl_createLayer() {
    Layer *pLayer = m_pViewer->getScene()->addLayer(1, "no type");
    QPointF pCenter = m_pViewer->getScene()->sceneRect().center();
    pLayer->shift(pCenter.x(), pCenter.y());
}
