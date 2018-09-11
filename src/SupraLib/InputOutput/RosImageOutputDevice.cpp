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
#include <supra_msgs/UInt8Image.h>

#include "RosImageOutputDevice.h"

#include <USImage.h>
#include <SyncRecordObject.h>
#include <utilities/Logging.h>
#include <utilities/utility.h>

#include <cassert>

namespace supra
{
	using namespace std;
	using namespace logging;

	RosImageOutputDevice::RosImageOutputDevice(tbb::flow::graph& graph, const std::string& nodeID, bool queueing)
		: AbstractOutput(graph, nodeID, queueing)
		, m_publisherNoImage(0)
	{
		m_valueRangeDictionary.set<string>("masterHostname", "localhost", "ROS Master");
		m_valueRangeDictionary.set<string>("topic", "image", "Image topic");
		m_callFrequency.setName("RosImageOut");
		m_isReady = false;

		configurationDone();
	}

	RosImageOutputDevice::~RosImageOutputDevice()
	{
		m_isReady = false;
	}

	void RosImageOutputDevice::initializeOutput()
	{
		log_info("RosImage topic: ", m_topic);

		m_rosWrapper = unique_ptr<RosWrapper>(new RosWrapper(m_masterHost));
		m_publisherNoImage = m_rosWrapper->advertise<supra_msgs::UInt8Image>(m_topic);

		m_isReady = true;
	}

	bool RosImageOutputDevice::ready()
	{
		return m_isReady;
	}

	void RosImageOutputDevice::startOutput()
	{
	}

	void RosImageOutputDevice::stopOutput()
	{
	}

	void RosImageOutputDevice::configurationDone()
	{
		m_topic = m_configurationDictionary.get<string>("topic");
		m_masterHost = m_configurationDictionary.get<string>("masterHostname");
	}

	void RosImageOutputDevice::writeData(std::shared_ptr<RecordObject> data)
	{
		if (m_isReady && getRunning())
		{
			m_callFrequency.measure();
			addData(data);
			m_callFrequency.measureEnd();
		}
	}

	void RosImageOutputDevice::addData(shared_ptr<const RecordObject> data)
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

	void RosImageOutputDevice::addSyncRecord(shared_ptr<const RecordObject> _syncMessage)
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
	void RosImageOutputDevice::addImageTemplated(shared_ptr<const USImage> _imageData)
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
				vec3s imageSize = imageData->getSize();
				supra_msgs::UInt8Image msg;
				msg.header.stamp.fromSec(imageData->getSyncTimestamp());
				msg.volume.resize(imageData->getData<ElementType>()->size());
				imageData->getData<ElementType>()->copyTo(&(msg.volume[0]), msg.volume.size());
				msg.width = imageSize.x;
				msg.height = imageSize.y;
				msg.depth = imageSize.z;

				m_rosWrapper->publish(msg, m_publisherNoImage);
			}
		}
	}

	void RosImageOutputDevice::addImage(shared_ptr<const RecordObject> _imageData)
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
				logging::log_error("RosImageOutputDevice: Image element type not supported");
				break;
			}
		}
	}
}
