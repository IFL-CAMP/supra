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
#include <utilities/RosWrapper.h>
#include <eden2020_msgs/Dataset.h>

#include "EdenImageOutputDevice.h"

#include <USImage.h>
#include <SyncRecordObject.h>
#include <utilities/Logging.h>
#include <utilities/utility.h>

#include <cassert>

namespace supra
{
	using namespace std;
	using namespace logging;

	EdenImageOutputDevice::EdenImageOutputDevice(tbb::flow::graph& graph, const std::string& nodeID, bool queueing)
		: AbstractOutput(graph, nodeID, queueing)
		, m_publisherNoImage(0)
	{
		m_valueRangeDictionary.set<string>("masterHostname", "localhost", "ROS Master");
		m_valueRangeDictionary.set<string>("topic", "image", "Image topic");
		m_valueRangeDictionary.set<double>("originOffsetX", 0, "Origin offset X");
		m_valueRangeDictionary.set<double>("originOffsetY", 0, "Origin offset Y");
		m_valueRangeDictionary.set<double>("originOffsetZ", 0, "Origin offset Z");
		m_callFrequency.setName("EdenImageOut");
		m_isReady = false;

		configurationDone();
	}

	EdenImageOutputDevice::~EdenImageOutputDevice()
	{
		m_isReady = false;
	}

	void EdenImageOutputDevice::initializeOutput()
	{
		log_info("EdenImage topic: ", m_topic);

		m_rosWrapper = unique_ptr<RosWrapper>(new RosWrapper(m_masterHost));
		m_publisherNoImage = m_rosWrapper->advertise<eden2020_msgs::Dataset>(m_topic);

		m_isReady = true;
	}

	bool EdenImageOutputDevice::ready()
	{
		return m_isReady;
	}

	void EdenImageOutputDevice::startOutput()
	{
	}

	void EdenImageOutputDevice::stopOutput()
	{
	}

	void EdenImageOutputDevice::configurationDone()
	{
		m_topic = m_configurationDictionary.get<string>("topic");
		m_masterHost = m_configurationDictionary.get<string>("masterHostname");
		m_originOffsetX = m_configurationDictionary.get<double>("originOffsetX");
		m_originOffsetY = m_configurationDictionary.get<double>("originOffsetY");
		m_originOffsetZ = m_configurationDictionary.get<double>("originOffsetZ");
	}

	void EdenImageOutputDevice::writeData(std::shared_ptr<RecordObject> data)
	{
		if (m_isReady && getRunning())
		{
			m_callFrequency.measure();
			addData(data);
			m_callFrequency.measureEnd();
		}
	}

	void EdenImageOutputDevice::addData(shared_ptr<const RecordObject> data)
	{
		switch (data->getType())
		{
		case TypeSyncRecordObject:
			addSyncRecord(data);
			break;
		case TypeUSImage:
			addImage(data);
			break;
		case TypeTrackerDataSet:
		case TypeRecordUnknown:
		default:
			break;
		}
	}

	void EdenImageOutputDevice::addSyncRecord(shared_ptr<const RecordObject> _syncMessage)
	{
		auto syncMessage = dynamic_pointer_cast<const SyncRecordObject>(_syncMessage);
		if (syncMessage)
		{
			auto mainRecord = syncMessage->getMainRecord();
			if (mainRecord->getType() == TypeUSImage)
			{
				addImage(mainRecord);
			}
		}
	}

	template <typename ElementType>
	void EdenImageOutputDevice::addImageTemplated(shared_ptr<const USImage> _imageData)
	{
		//TODO check for other element types
		auto imageData = dynamic_pointer_cast<const USImage>(_imageData);

		if (imageData)
		{
			auto properties = imageData->getImageProperties();
			if (
				properties->getImageType() == USImageProperties::BMode ||
				properties->getImageType() == USImageProperties::Doppler)
			{
				auto inImageData = imageData->getData<ElementType>();
				if (!inImageData->isHost() && !inImageData->isBoth())
				{
					inImageData = make_shared<Container<ElementType> >(LocationHost, *inImageData);
				}
				vec3s imageSize = imageData->getSize();
				eden2020_msgs::Dataset msg;
				msg.header.frame_id = "us_transducer";

				msg.header.stamp.fromSec(imageData->getSyncTimestamp());
				msg.origin.header.stamp = msg.header.stamp;
				msg.origin.header.frame_id = msg.header.frame_id;
				msg.volume.resize(inImageData->size());
				std::transform(inImageData->get(), inImageData->get() + msg.volume.size(), &(msg.volume[0]), [](const ElementType& x) -> int16_t { return (int16_t) x; });
				msg.width = imageSize.x;
				msg.height = imageSize.y;
				msg.depth = imageSize.z;

				msg.res_x = properties->getImageResolution();
				msg.res_y = properties->getImageResolution();
				msg.res_z = properties->getImageResolution();

				// This contains the change from "y forward" in supra to "z forward" in EDEN
				msg.origin.pose.position.x = - (double)(msg.width) * msg.res_x / 2.0;
				msg.origin.pose.position.y = (double)(msg.depth) * msg.res_y / 2.0;
				msg.origin.pose.position.z = 0;
				msg.origin.pose.position.x -= m_originOffsetX;
				msg.origin.pose.position.y -= m_originOffsetY;
				msg.origin.pose.position.z -= m_originOffsetZ;

				// This is a rotation of 90 degree around the x axis
				msg.origin.pose.orientation.x = sin(45.0 / 180.0 * M_PI);
				msg.origin.pose.orientation.y = 0;
				msg.origin.pose.orientation.z = 0;
				msg.origin.pose.orientation.w = sin(45.0 / 180.0 * M_PI);

				msg.uuid = "us_" + std::to_string(msg.header.stamp.sec) + "_" + std::to_string(msg.header.stamp.nsec);
				msg.source_uuid = msg.uuid;
				msg.modality = "us";

				m_rosWrapper->publish(msg, m_publisherNoImage);
			}
		}
	}

	void EdenImageOutputDevice::addImage(shared_ptr<const RecordObject> _imageData)
	{
		auto imageData = dynamic_pointer_cast<const USImage>(_imageData);
		if (imageData)
		{
			switch (imageData->getDataType())
			{
			case TypeUint8:
				addImageTemplated<uint8_t>(imageData);
				break;
			default:
				logging::log_error("EdenImageOutputDevice: Image element type not supported");
				break;
			}
		}
	}
}
