#include <QIOForm.h>
#include <ui_QIOForm.h>


IOForm::IOForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::IOForm)
{
    ui->setupUi(this);
}

IOForm::~IOForm()
{
    delete ui;
}
