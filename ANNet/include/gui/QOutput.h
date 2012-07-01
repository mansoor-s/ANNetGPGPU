/*
 * QOutput.h
 *
 *  Created on: 01.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#ifndef QOUTPUT_H_
#define QOUTPUT_H_

#include <Qt/QtGui>
#include <ANNet>


class Output: public QTableWidget {
public:
	Output(QWidget *parent = NULL);
	virtual ~Output();

	void Display(ANN::BPNet *pNet);
};

#endif /* QOUTPUT_H_ */
