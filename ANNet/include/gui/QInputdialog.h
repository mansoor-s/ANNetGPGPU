/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/

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
