// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "QImagePreviewReciever.h"

namespace supra
{
	QImagePreviewReciever::QImagePreviewReciever(QWidget* parentWidget, QVBoxLayout* targetLayout, const QString& name)
		:QWidget(parentWidget)
	{
		p_labelImage = new QLabel(this);

		targetLayout->addWidget(p_labelImage);

		p_labelImage->setVisible(false);
	}


	void QImagePreviewReciever::previewReadyImage(const std::shared_ptr<QImage> image)
	{
		p_labelImage->setVisible(true);
		p_labelImage->setPixmap(QPixmap::fromImage(*image));
		p_labelImage->adjustSize();
	}

	QImagePreviewReciever::~QImagePreviewReciever()
	{
		//p_labelImage->layout()->removeWidget(p_labelImage);
		delete p_labelImage;
	}
}