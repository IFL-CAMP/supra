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

#ifndef __PREVIEWBUILDERQT_H__
#define __PREVIEWBUILDERQT_H__

#include <memory>
#include <AbstractNode.h>
#include <QObject>
#include <QImage>
#include <QWidget>
#include <QLabel>
#include <QFormLayout>
#include <QProgressBar>
#include <QGroupBox>
#include <tbb/flow_graph.h>
#include <RecordObject.h>
#include <TrackerDataSet.h>
#include <USImage.h>
#include <utilities/Logging.h>

namespace supra
{
	using std::shared_ptr;

	class CampvisPreviewReciever;
	class QImagePreviewReciever;
	class QTrackerPreviewReciever;

	/** 
	 * A graph node to preview image and tracking streams in the GUI.
	 *
	 * This graph node allows to preview streams in the gui.
	 * It gets the input data via it's input graph node,
	 * prepares it for display (if applicable) and sends it to 
	 * \see QImagePreviewReceiver , \see QTragkerPreviewReceiver , or
	 * \see CampvisPreviewReciever as appropriate.
	 */
	class previewBuilderQT : public QObject, public AbstractNode
	{
		Q_OBJECT

	private:
		typedef tbb::flow::function_node<shared_ptr<RecordObject>, tbb::flow::continue_msg, tbb::flow::rejecting> inputNodeType;

	public:
		/// Constructor of the node. Beside the standard node parameters, it takes as parameters 
		/// Its parentWidget, the layout to attach to, the maximum image size and whether to use
		/// linear interpolation for rescaling
		previewBuilderQT(tbb::flow::graph & g, const std::string & nodeID, std::string name, QWidget* parentWidget, QVBoxLayout* targetLayout, QSize imageMaxSize, bool linearInterpolation);
		/// Changes the size of image previews
		void setPreviewSize(QSize imageMaxSize);
		/// Sets the use of linear interpolation for image scaling
		void setLinearInterpolation(bool linearInterpolation);
		/// Removes the preview widgets
		void removePreviews();

	private:
		void processRecordObject(const shared_ptr<RecordObject> inMessage);
		void addImagePreviewWidget();
		void addTrackingPreviewWidget();

		template <typename T>
		std::tuple<double, double, bool> computeWorldSize(std::shared_ptr < USImage<T> > image);

		bool m_haveImagePreview;
		bool m_haveTrackingPreview;
		QWidget* m_parentWidget;
		QVBoxLayout* m_targetLayout;
		std::string m_name;
		inputNodeType m_nodeIn;
		QSize m_imageMaxSize;
		bool m_linearInterpolation;

		size_t m_layerToShow;

#ifdef HAVE_CAMPVIS
		CampvisPreviewReciever* m_pCampvisPreview;
#endif
		QImagePreviewReciever* m_pQimagePreview;
		QTrackerPreviewReciever* m_pTrackerPreview;
	public:
		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 0; }

		virtual tbb::flow::receiver<std::shared_ptr<RecordObject> > *
			getInput(size_t index);

	signals:
		/// QT signal that is emitted, when an image preview is ready
		void previewReadyImage(const std::shared_ptr<QImage> image);
		/// QT signal that is emitted, when a \see RecordObject is received
		void previewReadyObject(const std::shared_ptr<RecordObject> object);
		/// QT signal that is emitted, when a \see TrackerDataSet is received
		void previewReadyTracking(const std::shared_ptr<TrackerDataSet> track);
	};
}

#endif //!__PREVIEWBUILDERQT_H__
