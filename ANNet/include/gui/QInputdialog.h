#ifndef INPUTDIALOG_H
#define INPUTDIALOG_H

#include <Qt/QtGui>


class InputDialog : public QWidget
{
    Q_OBJECT
    
public:
    explicit InputDialog(QWidget *parent = 0);
    ~InputDialog();
    
public slots:
    void sl_chooseInpContent();
    void sl_chooseOutContent();
    void sl_openTextFile(QString);

private:
};

#endif // INPUTDIALOG_H
