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

#ifndef __QTRACKERPREVIEWRECIEVER_H__
#define __QTRACKERPREVIEWRECIEVER_H__

#include <memory>
#include <QObject>
#include <QWidget>
#include <QLabel>
#include <QFormLayout>
#include <QProgressBar>
#include <QGroupBox>
#include <TrackerDataSet.h>

namespace supra
{
	/**
	 * Widget that provides a preview of a tracking stream
	 *
	 * Creates all the widgets used to display the data and
	 * receives the data from PreviewBuilderQT via signals.
	 */
	class QTrackerPreviewReciever : public QWidget
	{
		Q_OBJECT

	public:
		/**
		 * Constructor of QTrackerPreviewReciever
		 *
		 * \param parentWidget	Pointer to the widget that will be registered as parent of the new widget
		 * \param targetLayout	Pointer to the layout in which the widget will be embedded
		 * \param name			Name of the preview
		 */
		QTrackerPreviewReciever(QWidget* parentWidget, QVBoxLayout* targetLayout, const QString& name);
		~QTrackerPreviewReciever();

	private:
		QGroupBox* p_groupBox;
		QFormLayout* m_boxLayout;
		QVBoxLayout* m_targetLayout;
		std::vector<QProgressBar*> m_qualityBars;
		std::vector<QLabel*> m_labels;

	public slots:
		/// QT slot that triggers an update of the visualization with the passed TrackerDataSet
		void previewReadyTracking(const std::shared_ptr<TrackerDataSet> trackerSet);
	};
}
#endif //!__QTRACKERPREVIEWRECIEVER_H__
