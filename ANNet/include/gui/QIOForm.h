#ifndef QIOFORM_H
#define QIOFORM_H

#include <QWidget>

namespace Ui {
class IOForm;
}

class IOForm : public QWidget
{
    Q_OBJECT
    
public:
    explicit IOForm(QWidget *parent = 0);
    ~IOForm();
    
private:
    Ui::IOForm *ui;
};

#endif // QIOFORM_H
