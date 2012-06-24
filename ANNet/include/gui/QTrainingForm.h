#ifndef QTRAININGFORM_H
#define QTRAININGFORM_H

#include <QWidget>
#include <cstring>


namespace Ui {
class TrainingForm;
}

class TrainingForm : public QWidget
{
    Q_OBJECT
    
public:
    explicit TrainingForm(QWidget *parent = 0);
    ~TrainingForm();
    
    int getMaxCycles() const;
    float getMaxError() const;

    std::string getTransfFunct();

private:
    Ui::TrainingForm *ui;
};

#endif // QTRAININGFORM_H
