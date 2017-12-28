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

#include "previewBuilderQt.h"
#include "QImagePreviewReciever.h"
#include "QTrackerPreviewReciever.h"
#ifdef HAVE_CAMPVIS
#include "CampvisPreviewReciever.h"
#endif

namespace supra
{
	using namespace ::tbb::flow;
	using namespace ::std;

	previewBuilderQT::previewBuilderQT(graph & g, const std::string & nodeID, string name, string srcNodeID, size_t srcPort, QWidget* parentWidget, QVBoxLayout* targetLayout, QSize imageMaxSize, bool linearInterpolation)
		: QObject()
		, AbstractNode(nodeID)
		, m_nodeIn(g, 1,
			[this](const shared_ptr<RecordObject> & inMessage)
	{ processRecordObject(inMessage); })
		, m_imageMaxSize(imageMaxSize)
		, m_linearInterpolation(linearInterpolation)
		, m_haveImagePreview(false)
		, m_haveTrackingPreview(false)
		, m_parentWidget(parentWidget)
		, m_targetLayout(targetLayout)
		, m_name(name)
		, m_layerToShow(0)
		, m_srcNodeID(srcNodeID)
		, m_srcPort(srcPort)
	{
#ifdef HAVE_CAMPVIS
		m_pCampvisPreview = nullptr;
#endif
		m_pQimagePreview = nullptr;
		m_pTrackerPreview = nullptr;

		addTrackingPreviewWidget();
		addImagePreviewWidget();
	}

	void previewBuilderQT::removePreviews()
	{
		if (m_pTrackerPreview)
		{
			m_pTrackerPreview->setParent(nullptr);
			delete m_pTrackerPreview;
			m_pTrackerPreview = nullptr;
		}
		if (m_pQimagePreview)
		{
			m_pQimagePreview->setParent(nullptr);
			delete m_pQimagePreview;
			m_pQimagePreview = nullptr;
		}
#ifdef HAVE_CAMPVIS
		if (m_pCampvisPreview)
		{
			m_pCampvisPreview->setParent(nullptr);
			delete m_pCampvisPreview;
			m_pCampvisPreview = nullptr;
		}
#endif
	}

	void previewBuilderQT::setPreviewSize(QSize imageMaxSize)
	{
		m_imageMaxSize = imageMaxSize;
	}

	void previewBuilderQT::setLinearInterpolation(bool linearInterpolation)
	{
		m_linearInterpolation = linearInterpolation;
	}

	void previewBuilderQT::processRecordObject(const shared_ptr<RecordObject> inMessage)
	{
		if (inMessage)
		{
			if (inMessage->getType() == TypeUSImage)
			{
				auto inImage8Bit = std::dynamic_pointer_cast<USImage<uint8_t>>(inMessage);
				auto inImage16Bit = std::dynamic_pointer_cast<USImage<int16_t>>(inMessage);
				logging::log_error_if(!inImage8Bit && !inImage16Bit, "Error casting a record object to USImage, although the type was 'TypeUSImage'");
				if (inImage8Bit || inImage16Bit)
				{
					bool is2D;
					if (inImage16Bit)
					{
						is2D = inImage16Bit->getSize().z == 1;
					}
					if (inImage8Bit)
					{
						is2D = inImage8Bit->getSize().z == 1;
					}

					bool useCampVis = !is2D;
#ifdef HAVE_CAMPVIS
					if (useCampVis)
					{
						emit previewReadyObject(inMessage);
					}
#else
					useCampVis = false;
#endif
					if (!useCampVis)
					{
						shared_ptr<QImage> qtimage;
						tuple<double, double, bool> worldSize;
						if (inImage8Bit)
						{
							m_layerToShow = m_layerToShow % inImage8Bit->getSize().z;
							auto inImageData = inImage8Bit->getData();
							if (!inImageData->isHost())
							{
								inImageData = make_shared<Container<uint8_t> >(LocationHost, *inImageData);
							}
							auto numChannels = inImage8Bit->getNumChannels();
							if (numChannels == 1 || numChannels == 3 || numChannels == 4)
							{
								QImage::Format format = QImage::Format_Grayscale8;
								if (numChannels == 3)
								{
									format = QImage::Format_RGB888;
								}
								else if (numChannels == 4)
								{
									format = QImage::Format_RGB32;
								}

								qtimage = std::make_shared<QImage>(
									static_cast<int>(inImage8Bit->getSize().x),
									static_cast<int>(inImage8Bit->getSize().y),
									format);
								for (size_t row = 0; row < inImage8Bit->getSize().y; row++)
								{
									std::memcpy(qtimage->scanLine(static_cast<int>(row)),
										inImageData->get() + row*inImage8Bit->getSize().x*numChannels + m_layerToShow * inImage8Bit->getSize().x*numChannels*inImage8Bit->getSize().y,
										inImage8Bit->getSize().x * sizeof(uint8_t) * numChannels);
								}
							}
							else
							{
								qtimage = std::make_shared<QImage>(
									static_cast<int>(inImage8Bit->getSize().x),
									static_cast<int>(inImage8Bit->getSize().y),
									QImage::Format_RGB32);
								for (size_t row = 0; row < inImage8Bit->getSize().y; row++)
								{
									uchar* destRow = qtimage->scanLine(static_cast<int>(row));
									const uint8_t* srcRow = inImageData->get() + row*inImage8Bit->getSize().x*numChannels + m_layerToShow * inImage8Bit->getSize().x*numChannels*inImage8Bit->getSize().y;
									for (size_t col = 0; col < inImage8Bit->getSize().x; col++)
									{
										destRow[4 * col + 1] = srcRow[col * numChannels + 0];
										destRow[4 * col + 2] = srcRow[col * numChannels + 1];
										destRow[4 * col + 3] = srcRow[col * numChannels + 2];
									}
								}
							}
							worldSize = computeWorldSize(inImage8Bit);
						}
						if (inImage16Bit)
						{
							m_layerToShow = m_layerToShow % inImage16Bit->getSize().z;
							auto inImageData = inImage16Bit->getData();
							if (!inImageData->isHost())
							{
								inImageData = make_shared<Container<int16_t> >(LocationHost, *inImageData);
							}

							auto numChannels = inImage16Bit->getNumChannels();
							if (numChannels == 1 || numChannels == 3 || numChannels == 4)
							{
								QImage::Format format = QImage::Format_Grayscale8;
								if (numChannels == 3)
								{
									format = QImage::Format_RGB888;
								}
								else if (numChannels == 4)
								{
									format = QImage::Format_RGB32;
								}

								qtimage = std::make_shared<QImage>(
									static_cast<int>(inImage16Bit->getSize().x),
									static_cast<int>(inImage16Bit->getSize().y),
									format);
								for (size_t row = 0; row < inImage16Bit->getSize().y; row++)
								{
									uchar* destRow = qtimage->scanLine(static_cast<int>(row));
									const int16_t*srcRow = inImageData->get() + row*inImage16Bit->getSize().x*numChannels + m_layerToShow * inImage16Bit->getSize().x*numChannels*inImage16Bit->getSize().y;
									for (size_t col = 0; col < inImage16Bit->getSize().x * numChannels; col++)
									{
										destRow[col] = static_cast<uint8_t>(min(static_cast<double>(abs(srcRow[col])), 255.0));
									}
								}
							}
							else {
								qtimage = std::make_shared<QImage>(
									static_cast<int>(inImage16Bit->getSize().x),
									static_cast<int>(inImage16Bit->getSize().y),
									QImage::Format_RGB32);
								for (size_t row = 0; row < inImage16Bit->getSize().y; row++)
								{
									uchar* destRow = qtimage->scanLine(static_cast<int>(row));
									const int16_t* srcRow = inImageData->get() + row*inImage16Bit->getSize().x*numChannels + m_layerToShow * inImage16Bit->getSize().x*numChannels*inImage16Bit->getSize().y;
									for (size_t col = 0; col < inImage16Bit->getSize().x; col++)
									{
										destRow[4 * col + 0] = static_cast<uint8_t>(min(static_cast<double>(abs(srcRow[col * numChannels + 0])), 255.0));
										destRow[4 * col + 1] = static_cast<uint8_t>(min(static_cast<double>(abs(srcRow[col * numChannels + 1])), 255.0));
										destRow[4 * col + 2] = static_cast<uint8_t>(min(static_cast<double>(abs(srcRow[col * numChannels + 2])), 255.0));
										destRow[4 * col + 3] = static_cast<uint8_t>(min(static_cast<double>(abs(srcRow[col * numChannels + 3])), 255.0));
									}
								}
							}

							worldSize = computeWorldSize(inImage16Bit);
						}
						m_layerToShow++;

						double worldWidth = get<0>(worldSize);
						double worldHeight = get<1>(worldSize);
						bool keepRatio = get<2>(worldSize);

						int imageWidth;
						int imageHeight;

						if (keepRatio)
						{
							if ((worldWidth / worldHeight) > (static_cast<double>(m_imageMaxSize.width()) / m_imageMaxSize.height()))
							{
								imageWidth = m_imageMaxSize.width();
								imageHeight = m_imageMaxSize.width() / worldWidth*worldHeight;
							}
							else {
								imageWidth = m_imageMaxSize.height() / worldHeight*worldWidth;
								imageHeight = m_imageMaxSize.height();
							}
						}
						else {
							imageWidth = m_imageMaxSize.width();
							imageHeight = m_imageMaxSize.height();
						}

						*qtimage = qtimage->scaled(
							imageWidth,
							imageHeight,
							Qt::IgnoreAspectRatio,
							(m_linearInterpolation ? Qt::SmoothTransformation : Qt::FastTransformation));

						emit previewReadyImage(qtimage);
					}
				}
			}
			else if (inMessage->getType() == TypeTrackerDataSet)
			{
				auto inTrackerData = std::dynamic_pointer_cast<TrackerDataSet>(inMessage);
				logging::log_error_if(!inTrackerData, "Error casting a record object to TrackerDataSet, although the type was 'TypeTrackerDataSet'");
				if (inTrackerData)
				{
					emit previewReadyTracking(inTrackerData);
				}
			}
		}
	}

	void previewBuilderQT::addImagePreviewWidget()
	{
		if (!m_haveImagePreview)
		{
#ifdef HAVE_CAMPVIS
			m_pCampvisPreview = new CampvisPreviewReciever(m_parentWidget, m_targetLayout, m_name);
			qRegisterMetaType<shared_ptr<RecordObject>>("std::shared_ptr<RecordObject>");
			connect(this, SIGNAL(previewReadyObject(const std::shared_ptr<RecordObject>)), m_pCampvisPreview, SLOT(previewReadyImage(const std::shared_ptr<RecordObject>)));
#endif
			//create the QImagePreviewReciever
			m_pQimagePreview = new QImagePreviewReciever(m_parentWidget, m_targetLayout, QString::fromStdString(m_name));
			qRegisterMetaType<shared_ptr<QImage>>("std::shared_ptr<QImage>");
			connect(this, SIGNAL(previewReadyImage(const std::shared_ptr<QImage>)), m_pQimagePreview, SLOT(previewReadyImage(const std::shared_ptr<QImage>)));

			m_haveImagePreview = true;
		}
	}

	void previewBuilderQT::addTrackingPreviewWidget()
	{
		if (!m_haveTrackingPreview)
		{
			//create the QImagePreviewReciever
			m_pTrackerPreview = new QTrackerPreviewReciever(m_parentWidget, m_targetLayout, QString::fromStdString(m_name));

			qRegisterMetaType<shared_ptr<TrackerDataSet>>("std::shared_ptr<TrackerDataSet>");
			connect(this, SIGNAL(previewReadyTracking(const std::shared_ptr<TrackerDataSet>)), m_pTrackerPreview, SLOT(previewReadyTracking(const std::shared_ptr<TrackerDataSet>)));
			m_haveTrackingPreview = true;
		}
	}

	template <typename T>
	std::tuple<double, double, bool> previewBuilderQT::computeWorldSize(std::shared_ptr < USImage<T> > image)
	{
		double worldWidth;
		double worldHeight;
		bool keepRatio;
		if (image->getImageProperties()->getImageState() == USImageProperties::ImageState::Scan)
		{
			double resolution = image->getImageProperties()->getImageResolution();
			worldWidth = image->getSize().x * resolution;
			worldHeight = image->getSize().y * resolution;
			keepRatio = true;
		}
		else {
			worldWidth = image->getImageProperties()->getNumScanlines() - 1;
			worldHeight = image->getImageProperties()->getDepth();
			keepRatio = false;
		}
		return make_tuple(worldWidth, worldHeight, keepRatio);
	}

	tbb::flow::receiver<std::shared_ptr<RecordObject> > *
		previewBuilderQT::getInput(size_t index) {
		if (index == 0)
		{
			return &m_nodeIn;
		}
		else
		{
			return nullptr;
		}
	};
}