/*
 * QTrainingThread.cpp
 *
 *  Created on: 08.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <gui/QTrainingThread.h>


TrainingThread::TrainingThread(ANN::BPNet *pNet, int iCycles, float fError, QObject *parent) : QThread(parent) {
	m_pNet 		= pNet;
	m_iCycles 	= iCycles;
	m_fError 	= fError;
}

TrainingThread::~TrainingThread() {
	// TODO Auto-generated destructor stub
}

void TrainingThread::run() {
	if(m_pNet != NULL) {
		m_fErrors = m_pNet->TrainFromData(m_iCycles, m_fError);
	}
	exec();
}

std::vector<float> TrainingThread::getErrors() const {
	return m_fErrors;
}
