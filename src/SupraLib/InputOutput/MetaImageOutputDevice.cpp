// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2011-2016, all rights reserved,
//      Christoph Hennersperger 
//		EmaiL christoph.hennersperger@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
//	and
//		Rüdiger Göbl
//		Email r.goebl@tum.de
//
// ================================================================================================

#include "MetaImageOutputDevice.h"
#include "MhdSequenceWriter.h"
#include <SyncRecordObject.h>
#include <USImage.h>
#include <Beamformer/USRawData.h>
#include <TrackerDataSet.h>
#include <utilities/Logging.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>
#include <cassert>

using namespace std;
namespace supra
{
	using namespace logging;

	MetaImageOutputDevice::MetaImageOutputDevice(tbb::flow::graph& graph, const std::string & nodeID)
		: AbstractOutput(graph, nodeID)
		, m_filename("output")
		, m_createSequences(false)
		, m_isRecording(false)
		, m_sequencesWritten(0)
		, m_active(true)
		, m_pWriter(nullptr)
	{
		m_callFrequency.setName("MHD");

		m_valueRangeDictionary.set<string>("filename", "output", "Filename");
		m_valueRangeDictionary.set<bool>("createSequences", { false, true }, true, "Sequences");
		m_valueRangeDictionary.set<bool>("active", { false, true }, true, "Active");

		m_isReady = false;
	}

	MetaImageOutputDevice::~MetaImageOutputDevice()
	{
		m_isReady = false;

		m_pWriter->closeWhenEverythingWritten();
	}

	void MetaImageOutputDevice::initializeOutput()
	{
		if (!m_createSequences)
		{
			initializeSequence();
			m_isReady = m_pWriter->isOpen();
		}
		else
		{
			m_isReady = true;
		}
	}

	bool MetaImageOutputDevice::ready()
	{
		return m_isReady;
	}

	void MetaImageOutputDevice::startOutput()
	{
	}

	void MetaImageOutputDevice::stopOutput()
	{
	}

	void MetaImageOutputDevice::startSequence()
	{
		if (m_createSequences)
		{
			lock_guard<mutex> l(m_writerMutex);
			initializeSequence();
			m_isRecording = true;
		}
	}

	void MetaImageOutputDevice::stopSequence()
	{
		if (m_createSequences)
		{
			lock_guard<mutex> l(m_writerMutex);
			m_isRecording = false;
			m_pWriter->closeWhenEverythingWritten();
			m_pWriter = nullptr;

			m_sequencesWritten++;
		}
	}

	void MetaImageOutputDevice::configurationDone()
	{
		m_filename = m_configurationDictionary.get<string>("filename");
		m_createSequences = m_configurationDictionary.get<bool>("createSequences");
		m_active = m_configurationDictionary.get<bool>("active");
	}

	void MetaImageOutputDevice::writeData(std::shared_ptr<RecordObject> data)
	{
		lock_guard<mutex> l(m_writerMutex);
		if (m_isReady && m_isRecording && getRunning())
		{
			m_callFrequency.measure();
			addData(data);
			m_callFrequency.measureEnd();
		}
	}

	void MetaImageOutputDevice::initializeSequence()
	{
		string filename = m_filename;
		if (m_createSequences)
		{
			filename += "_" + std::to_string(m_sequencesWritten);
		}
		log_info("MetaImage file Name: ", filename);

		if (m_pWriter)
		{
			m_pWriter->closeWhenEverythingWritten();
		}
		m_pWriter = new MhdSequenceWriter();
		m_pWriter->open(filename);
		m_isRecording = m_pWriter->isOpen();
	}

	void MetaImageOutputDevice::addData(shared_ptr<const RecordObject> data)
	{
		switch (data->getType())
		{
		case TypeSyncRecordObject:
			addSyncRecord(data);
			break;
		case TypeUSImage:
			addImage(data);
			break;
		case TypeUSRawData:
			addUSRawData(data);
			break;
		case TypeTrackerDataSet:
		case TypeRecordUnknown:
		default:
			break;
		}
	}

	void MetaImageOutputDevice::addSyncRecord(shared_ptr<const RecordObject> _syncMessage)
	{
		auto syncMessage = dynamic_pointer_cast<const SyncRecordObject>(_syncMessage);
		if (syncMessage)
		{
			auto mainRecord = syncMessage->getMainRecord();
			if (mainRecord->getType() == TypeUSImage)
			{
				auto successframeNum = addImage(mainRecord);
				bool success = get<0>(successframeNum);
				size_t frameNum = get<1>(successframeNum);

				if (success)
				{
					for (shared_ptr<const RecordObject> syncedO : syncMessage->getSyncedRecords())
					{
						if (syncedO->getType() == TypeTrackerDataSet)
						{
							addTracking(syncedO, frameNum);
						}
					}
				}
			}
		}
	}

	template <typename ElementType>
	std::pair<bool, size_t> MetaImageOutputDevice::addImageTemplated(shared_ptr<const USImage<ElementType> > imageData)
	{
		bool success = false;
		size_t frameNum = 0;
		if (imageData)
		{
			auto properties = imageData->getImageProperties();
			if (
				properties->getImageType() == USImageProperties::BMode ||
				properties->getImageType() == USImageProperties::Doppler)
			{
				double resolution = properties->getImageResolution();
				vec3s imageSize = imageData->getSize();
				if (imageSize.z == 0)
				{
					imageSize.z = 1;
				}

				shared_ptr<Container<ElementType> > pDataCopy;
				shared_ptr<const Container<ElementType> > pData;
				if (imageData->getData()->isHost())
				{
					pData = imageData->getData();
				}
				else {
					pDataCopy = make_shared<Container<ElementType> >(LocationHost, *(imageData->getData()));
					pData = pDataCopy;
				}

				auto successframeNum = m_pWriter->addImage(
					pData->get(), imageSize.x, imageSize.y, imageSize.z,
					imageData->getSyncTimestamp(), resolution,
					[pData](const uint8_t*, size_t){}
				);
				success = get<0>(successframeNum);
				frameNum = get<1>(successframeNum);
			}
		}
		return std::make_pair(success, frameNum);
	}

	template <typename ElementType>
	std::pair<bool, size_t> MetaImageOutputDevice::addUSRawDataTemplated(shared_ptr<const USRawData<ElementType> > rawData)
	{
		bool success = false;
		size_t frameNum = 0;
		if (rawData)
		{
			size_t numChannels = rawData->getNumReceivedChannels();
			size_t numSamples = rawData->getNumSamples();
			size_t numScanlines = rawData->getNumScanlines();

			shared_ptr<Container<ElementType> > pDataCopy;
			shared_ptr<const Container<ElementType> > pData;
			if (rawData->getData()->isHost())
			{
				pData = rawData->getData();
			}
			else {
				pDataCopy = make_shared<Container<int16_t> >(LocationHost, *(rawData->getData()));
				pData = pDataCopy;
			}
			double resolution = rawData->getImageProperties()->getSampleDistance();
			auto successframeNum = m_pWriter->addImage<ElementType>(
				pData->get(), numSamples, numChannels, numScanlines, 
				rawData->getSyncTimestamp(), resolution,
				[pData](const uint8_t*, size_t){}
			);
			success = get<0>(successframeNum);
			frameNum = get<1>(successframeNum);
		}
		return std::make_pair(success, frameNum);
	}

	std::pair<bool, size_t> MetaImageOutputDevice::addImage(shared_ptr<const RecordObject> _imageData)
	{
		std::pair<bool, size_t> successframeNum = std::make_pair(false, 0);
		auto imageData8Bit = dynamic_pointer_cast<const USImage<uint8_t>>(_imageData);
		if (imageData8Bit)
		{
			successframeNum = addImageTemplated<uint8_t>(imageData8Bit);
		}

		auto imageData16Bit = dynamic_pointer_cast<const USImage<int16_t>>(_imageData);
		if (imageData16Bit)
		{
			successframeNum = addImageTemplated<int16_t>(imageData16Bit);
		}
		return successframeNum;
	}

	std::pair<bool, size_t> MetaImageOutputDevice::addUSRawData(shared_ptr<const RecordObject> _rawData)
	{
		std::pair<bool, size_t> successframeNum = std::make_pair(false, 0);
		auto rawData = dynamic_pointer_cast<const USRawData<int16_t>>(_rawData);
		if (rawData)
		{
			successframeNum = addUSRawDataTemplated<int16_t>(rawData);
		}
		return successframeNum;
	}

	void MetaImageOutputDevice::addTracking(shared_ptr<const RecordObject> _trackData, size_t frameNum)
	{
		double maxAcceptanceQual = 700;

		auto trackData = dynamic_pointer_cast<const TrackerDataSet>(_trackData);
		if (trackData)
		{
			for (auto t : trackData->getSensorData())
			{
				m_pWriter->addTracking(frameNum, t.getMatrix(), t.getQuality() < maxAcceptanceQual, std::to_string(t.getUID()));
			}
		}
	}
}