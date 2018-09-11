#include "previewWidget.h"
#include "ui_previewWidget.h"

previewWidget::previewWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::previewWidget)
{
    ui->setupUi(this);
}

previewWidget::~previewWidget()
{
    delete ui;
}
