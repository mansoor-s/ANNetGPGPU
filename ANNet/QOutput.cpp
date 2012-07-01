/*
 * QOutput.cpp
 *
 *  Created on: 01.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <gui/QOutput.h>
#include <containers/ANTrainingSet.h>


Output::Output(QWidget *parent) : QTableWidget(parent) {
	setAlternatingRowColors(true);
}

Output::~Output() {
	// TODO Auto-generated destructor stub
}

void Output::Display(ANN::BPNet *pNet) {
	QFont font;
	font.setBold(true);
	QTableWidgetItem *pItem;

	setColumnCount(pNet->GetTrainingSet()->GetNrElements()*2);
	setRowCount(pNet->GetOPLayer()->GetNeurons().size() );
	for(unsigned int i = 0; i < pNet->GetOPLayer()->GetNeurons().size(); i++) {
		pItem = new QTableWidgetItem("Neuron "+QString::number(i+1));
		pItem->setFont(font);
		setVerticalHeaderItem(i, pItem);
	}

	for(unsigned int i = 0; i < pNet->GetTrainingSet()->GetNrElements(); i++) {
		pItem = new QTableWidgetItem("Wished\nfrom set: "+QString::number(i+1));
		pItem->setFont(font);
		setHorizontalHeaderItem(2*i, pItem);

		pItem = new QTableWidgetItem("Achieved\nfrom set: "+QString::number(i+1));
		pItem->setFont(font);
		setHorizontalHeaderItem(2*i+1, pItem);

		pNet->SetInput(pNet->GetTrainingSet()->GetInput(i) );
		pNet->PropagateFW();

		std::vector<float> vOut = pNet->GetTrainingSet()->GetOutput(i);
		for(unsigned int j = 0; j < vOut.size(); j++) {
			setItem(j, 2*i, new QTableWidgetItem(QString::number(vOut.at(j))) );
		}
		vOut = pNet->GetOutput();
		for(unsigned int j = 0; j < vOut.size(); j++) {
			setItem(j, 2*i+1, new QTableWidgetItem(QString::number(vOut.at(j))) );
		}
	}

}
