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

#ifndef __CAMPVISPREVIEWRECIEVER_H__
#define __CAMPVISPREVIEWRECIEVER_H__

#include <memory>
#include <atomic>
#include <QObject>
#include <QImage>
#include <QWidget>
#include <QLabel>
#include <QFormLayout>

#include "RecordObject.h"

//campvis
#include <core/datastructures/datacontainer.h>
#include <ext/cgt/qt/qtthreadedcanvas.h>

namespace campvis{
	class VolumeExplorerPipeline;
}

namespace supra
{
	//forward declarations
	class USImage;

	/**
	 * Widget that employs CAMPvis to visualize volumes
	 *
	 * Creates all the widgets used to display the data and
	 * receives the data from PreviewBuilderQT via signals.
	 */
	class CampvisPreviewReciever : public QWidget
	{
		Q_OBJECT

	public:
		/**
		 * Constructor of CampvisPreviewReciever
		 * 
		 * \param parentWidget	Pointer to the widget that will be registered as parent of the new widget
		 * \param targetLayout	Pointer to the layout in which the widget will be embedded
		 * \param name			Name of the preview
		 */
		CampvisPreviewReciever(QWidget* parentWidget, QVBoxLayout* targetLayout, const std::string& name);
		~CampvisPreviewReciever();

	private:
		template <typename ElementType>
		void previewReadyImageTemplated(const std::shared_ptr<USImage> image);

		std::string m_name;
		std::unique_ptr<campvis::VolumeExplorerPipeline> m_pipeline;
		std::unique_ptr<campvis::DataContainer> m_campvisDatacontainer;
		cgt::QtThreadedCanvas* m_canvas;

		static void initializeCampvis();

		static std::unique_ptr<cgt::QtThreadedCanvas> m_campvisBackgroundCanvas;
		static std::atomic<bool> m_campvisInitialized;


	public slots:
		/// QT slot that triggers an update of the visualization with the passed image
		void previewReadyImage(const std::shared_ptr<RecordObject> image);
	};
}
#endif //!__CAMPVISPREVIEWRECIEVER_H__
