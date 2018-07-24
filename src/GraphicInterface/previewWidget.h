#ifndef PREVIEWWIDGET_H
#define PREVIEWWIDGET_H

#include <QWidget>

namespace Ui {
class previewWidget;
}

class previewWidget : public QWidget
{
    Q_OBJECT

public:
    explicit previewWidget(QWidget *parent = 0);
    ~previewWidget();

    Ui::previewWidget *ui;
};

#endif // PREVIEWWIDGET_H
