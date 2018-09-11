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

#include "QTrackerPreviewReciever.h"
#include <utilities/utility.h>
#include <cmath>

namespace supra
{
	QTrackerPreviewReciever::QTrackerPreviewReciever(QWidget* parentWidget, QVBoxLayout* targetLayout, const QString& name)
		:QWidget(parentWidget)
		, m_targetLayout(targetLayout)
	{
		p_groupBox = new QGroupBox(this);
		m_boxLayout = new QFormLayout(p_groupBox);
		p_groupBox->setTitle(name);
		m_boxLayout->setSpacing(6);
		m_boxLayout->setContentsMargins(11, 11, 11, 11);

		m_targetLayout->addWidget(p_groupBox);

		p_groupBox->setVisible(false);
	}

	void QTrackerPreviewReciever::previewReadyTracking(const std::shared_ptr<TrackerDataSet> trackerSet)
	{
		p_groupBox->setVisible(true);
		std::vector<TrackerData> t = trackerSet->getSensorData();

		double maxAcceptanceQual = 700;

		if (t.size() > m_qualityBars.size())
		{
			//add addtional bars
			QProgressBar* newBar = new QProgressBar(this);
			newBar->setMaximum(maxAcceptanceQual);
			newBar->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
			newBar->setTextVisible(false);
			QLabel* newLabel = new QLabel("Pos:");
			m_boxLayout->addRow(newLabel, newBar);
			m_qualityBars.push_back(newBar);
			m_labels.push_back(newLabel);
		}
		for (size_t i = 0; i < t.size(); i++)
		{
			auto matrix = t[i].getMatrix();

			double roll = std::atan2(matrix[4], matrix[0]) / M_PI*180.0;
			double pitch = std::atan2(-matrix[8], std::sqrt(matrix[0] * matrix[0] + matrix[4] * matrix[4])) / M_PI*180.0;
			double yaw = std::atan2(matrix[9], matrix[10]) / M_PI*180.0;

			//set text
			m_labels[i]->setText(
				QString("Pos: %1 %2 %3 Rot: r %4 p %5 y %6").
				arg(matrix[3], 5, 'f', 2).
				arg(matrix[7], 5, 'f', 2).
				arg(matrix[11], 5, 'f', 2).
				arg(roll, 5, 'f', 2).
				arg(pitch, 5, 'f', 2).
				arg(yaw, 5, 'f', 2));
			m_qualityBars[i]->setValue(
				(t[i].getQuality() > maxAcceptanceQual ?
					0 : t[i].getQuality()));
		}
	}

	QTrackerPreviewReciever::~QTrackerPreviewReciever()
	{
		delete p_groupBox;
	}
}
