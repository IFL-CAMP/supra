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

#ifndef __QIMAGEPREVIEWRECIEVER_H__
#define __QIMAGEPREVIEWRECIEVER_H__

#include <memory>
#include <QObject>
#include <QImage>
#include <QWidget>
#include <QLabel>
#include <QFormLayout>

namespace supra
{
	/**
	 * Widget that uses QT images to visualize volumes
	 *
	 * Creates all the widgets used to display the data and
	 * receives the data from PreviewBuilderQT via signals.
	 */
	class QImagePreviewReciever : public QWidget
	{
		Q_OBJECT

	public:
		/**
		 * Constructor of QImagePreviewReciever
		 *
		 * \param parentWidget	Pointer to the widget that will be registered as parent of the new widget
		 * \param targetLayout	Pointer to the layout in which the widget will be embedded
		 * \param name			Name of the preview
		 */
		QImagePreviewReciever(QWidget* parentWidget, QVBoxLayout* targetLayout, const QString& name);
		~QImagePreviewReciever();

	private:
		QLabel* p_labelImage;

	public slots:
		/// QT slot that triggers an update of the visualization with the passed image
		void previewReadyImage(const std::shared_ptr<QImage> image);
	};
}

#endif //!__QIMAGEPREVIEWRECIEVER_H__
