#include <QTrainingForm.h>
#include <ui_QTrainingForm.h>


TrainingForm::TrainingForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TrainingForm)
{
    ui->setupUi(this);
}

TrainingForm::~TrainingForm()
{
    delete ui;
}

int TrainingForm::getMaxCycles() const {
	return ui->m_SBMax->value();
}

float TrainingForm::getMaxError() const {
	float fVal = 10000.0;
	return ((float)ui->m_SBError->value())/fVal;
}

std::string TrainingForm::getTransfFunct() {
	return ui->m_CBTransferFunction->currentText().toStdString();
}
