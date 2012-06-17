#include <gui/QInputdialog.h>

InputDialog::InputDialog(QWidget *parent) :
    QWidget(parent)
{
}

InputDialog::~InputDialog()
{
}

void InputDialog::sl_chooseInpContent() {
    QStringList files = QFileDialog::getOpenFileNames(
                            this,
                            "Select one or more files to open",
                            "/home",
                            "Textfiles (*.txt)");
}

void InputDialog::sl_chooseOutContent() {
    QStringList files = QFileDialog::getOpenFileNames(
                            this,
                            "Select one or more files to open",
                            "/home",
                            "Textfiles (*.txt)");
}

void InputDialog::sl_openTextFile(QString sFile) {
    QFile file(sFile);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
    }
}
