#include <iostream>
#include <gui/QLayer.h>
#include <gui/QMainWindow.h>
#include <gui/utils/stylehelper.h>
#include <gui/utils/manhattanstyle.h>  //"manhattanstyle.h"
#include <math/ANFunctions.h>
#include <containers/ANTrainingSet.h>


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
    createGraph();
}

MainWindow::~MainWindow()
{
    
}

void MainWindow::createGraph() {
	// give the axes some labels:
	m_pCustomPlot->xAxis->setLabel("Training cycle (t)");
	m_pCustomPlot->yAxis->setLabel("Standard deviation (SE)");
	// set axes ranges, so we see all data:
	m_pCustomPlot->xAxis->setRange(0, 1);
	m_pCustomPlot->yAxis->setRange(0, 10);
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
	  float fInp1[3];
	  fInp1[0] = 0;
	  fInp1[1] = 0;
	  fInp1[2] = 0;

	  float fInp2[3];
	  fInp2[0] = 0;
	  fInp2[1] = 1;
	  fInp2[2] = 0;

	  float fInp3[3];
	  fInp3[0] = 0;
	  fInp3[1] = 0;
	  fInp3[2] = 1;

	  float fInp4[3];
	  fInp4[0] = 1;
	  fInp4[1] = 0;
	  fInp4[2] = 1;

	  float fOut1[6];
	  fOut1[0] = 0.1;
	  fOut1[1] = 0.2;
	  fOut1[2] = 0.3;
	  fOut1[3] = 0.4;
	  fOut1[4] = 0.5;
	  fOut1[5] = 0.6;
	  float fOut2[6];

	  fOut2[0] = 0;
	  fOut2[1] = 1;
	  fOut2[2] = 0;
	  fOut2[3] = 0;
	  fOut2[4] = 0;
	  fOut2[5] = 0;

	  float fOut3[6];
	  fOut3[0] = 0;
	  fOut3[1] = 0;
	  fOut3[2] = 1;
	  fOut3[3] = 0;
	  fOut3[4] = 0;
	  fOut3[5] = 0;

	  float fOut4[6];
	  fOut4[0] = 0;
	  fOut4[1] = 0;
	  fOut4[2] = 0;
	  fOut4[3] = 1;
	  fOut4[4] = 0;
	  fOut4[5] = 0;

	  ANN::TrainingSet input;
	  input.AddInput(fInp1, 3);
	  input.AddOutput(fOut1, 6);
	  input.AddInput(fInp2, 3);
	  input.AddOutput(fOut2, 6);
	  input.AddInput(fInp3, 3);
	  input.AddOutput(fOut3, 6);
	  input.AddInput(fInp4, 3);
	  input.AddOutput(fOut4, 6);
/////////////////////////////////////////////

	int iCycles 			= m_pTrainingDial->getMaxCycles();
	float fMaxError 		= m_pTrainingDial->getMaxError();
	std::string sTFunct 	= m_pTrainingDial->getTransfFunct().data();

	std::cout<<"soft coded: "<<std::endl;

	ANN::BPNet *pNet = m_pViewer->getScene()->getANNet();
	pNet->SetLearningRate(0.2);
	pNet->SetMomentum(0.9);
	pNet->SetWeightDecay(0);
	pNet->SetTransfFunction(ANN::Functions::ResolveTransfFByName(sTFunct.data()));

	pNet->SetTrainingSet(input);
	m_vErrors = pNet->TrainFromData(iCycles, 0.001);
	std::cout<<pNet<<std::endl;

	iCycles = m_vErrors.size();
	if(pNet == NULL || iCycles == 0)
		return;
	else {
		// generate some data:
		float fGreatest = m_vErrors[0];
		QVector<double> x(iCycles), y(iCycles); // initialize with entries 0..100
		for (int i=0; i < iCycles; i++) {
			x[i] = i;
			y[i] = m_vErrors[i];
			if(fGreatest < m_vErrors[i])
				fGreatest = m_vErrors[i];
		}

		// set axes ranges, so we see all data:
		m_pCustomPlot->xAxis->setRange(0, iCycles);
		m_pCustomPlot->yAxis->setRange(0, fGreatest);
		// create graph and assign data to it:
		m_pCustomPlot->addGraph();
		m_pCustomPlot->graph(0)->setData(x, y);
		m_pCustomPlot->graph(0)->setBrush(QBrush(QColor(0, 0, 255, 20))); // first graph will be filled with translucent blue
		m_pCustomPlot->replot();
	}
}

void MainWindow::sl_createLayer() {
    Layer *pLayer = m_pViewer->getScene()->addLayer(1, "no type");
    QPointF pCenter = m_pViewer->getScene()->sceneRect().center();
    pLayer->shift(pCenter.x(), pCenter.y());
}
