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

	MetaImageOutputDevice::MetaImageOutputDevice(tbb::flow::graph& graph, const std::string & nodeID, bool queueing)
		: AbstractOutput(graph, nodeID, queueing)
		, m_filename("output")
		, m_createSequences(false)
		, m_isRecording(false)
		, m_sequencesWritten(0)
		, m_active(true)
		, m_pWriter(nullptr)
		, m_lastElementNumber(0)
	{
		m_callFrequency.setName("MHD");

		m_valueRangeDictionary.set<string>("filename", "output", "Filename");
		m_valueRangeDictionary.set<bool>("createSequences", { false, true }, true, "Sequences");
		m_valueRangeDictionary.set<bool>("active", { false, true }, true, "Active");
		m_valueRangeDictionary.set<uint32_t>("maxElements", 1, std::numeric_limits<uint32_t>::max(), 10000, "Maximum elements");

		m_isReady = false;
	}

	MetaImageOutputDevice::~MetaImageOutputDevice()
	{
		m_isReady = false;

		if (m_pWriter)
		{
			m_pWriter->closeWhenEverythingWritten(true);
		}
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
		startSequence();
	}

	void MetaImageOutputDevice::stopOutput()
	{
		stopSequence();
	}

	void MetaImageOutputDevice::startSequence()
	{
		//std::cout << "Entered meta image startSequence" << std::endl;
		if (m_createSequences)
		{
			//std::cout << "Initializing meta image sequence" << std::endl;
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
		m_maxElementNumber = m_configurationDictionary.get<uint32_t>("maxElements")-1;
	}

	void MetaImageOutputDevice::writeData(std::shared_ptr<RecordObject> data)
	{
		lock_guard<mutex> l(m_writerMutex);
		typedef std::chrono::high_resolution_clock Clock;
		typedef std::chrono::milliseconds milliseconds;
		Clock::time_point t0 = Clock::now();
		if (m_isReady && m_isRecording && getRunning())
		{
			m_callFrequency.measure();
			addData(data);
			m_callFrequency.measureEnd();
		}
		Clock::time_point t1 = Clock::now();
		milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
		std::cout << "Time to write data: " << ms.count() << "ms\n";
	}

	void MetaImageOutputDevice::initializeSequence()
	{
		string filename = m_filename;
		if (m_createSequences)
		{
			filename += "_" + std::to_string(m_sequencesWritten);
		}
		//log_info("MetaImage file Name: ", filename);
		std::cout << "output file: " << filename << std::endl;

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
		if (m_lastElementNumber < m_maxElementNumber)
		{
			std::pair<bool, size_t> elementAddSuccess(false,0);

			switch (data->getType())
			{
			case TypeSyncRecordObject:
				elementAddSuccess = addSyncRecord(data);
				break;
			case TypeUSImage:
				elementAddSuccess = addImage(data);
				break;
			case TypeUSRawData:
				elementAddSuccess = addUSRawData(data);
				break;
			case TypeTrackerDataSet:
			case TypeRecordUnknown:
			default:
				break;
			}

			if (elementAddSuccess.first)
			{
				m_lastElementNumber = elementAddSuccess.second;
			}
		}
		else {
			logging::log_warn("Could not write frame to MHD " + m_filename + ", file already contains " + std::to_string(m_lastElementNumber+1) + " frames.");
		}
	}

	std::pair<bool, size_t> MetaImageOutputDevice::addSyncRecord(shared_ptr<const RecordObject> _syncMessage)
	{
		bool success = false;
		size_t frameNum = 0;

		auto syncMessage = dynamic_pointer_cast<const SyncRecordObject>(_syncMessage);
		if (syncMessage)
		{
			auto mainRecord = syncMessage->getMainRecord();
			if (mainRecord->getType() == TypeUSImage)
			{
				auto successframeNum = addImage(mainRecord);
				success = get<0>(successframeNum);
				frameNum = get<1>(successframeNum);

				if (success)
					for (shared_ptr<const RecordObject> syncedO : syncMessage->getSyncedRecords())
						if (syncedO->getType() == TypeTrackerDataSet)
							addTracking(syncedO, frameNum);
			}
		}

		return std::make_pair(success, frameNum);
	}

	template <typename ElementType>
	std::pair<bool, size_t> MetaImageOutputDevice::addImageTemplated(shared_ptr<const USImage> imageData)
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
				if (imageData->getData<ElementType>()->isHost())
				{
					pData = imageData->getData<ElementType>();
				}
				else {
					pDataCopy = make_shared<Container<ElementType> >(LocationHost, *(imageData->getData<ElementType>()));
					pData = pDataCopy;
				}

				std::cout << "output image data size: " << imageSize.x << " " << imageSize.y << " " << imageSize.z << std::endl;
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
	std::pair<bool, size_t> MetaImageOutputDevice::addUSRawDataTemplated(shared_ptr<const USRawData> rawData)
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
			if (rawData->getData<ElementType>()->isHost())
			{
				pData = rawData->getData<ElementType>();
			}
			else {
				pDataCopy = make_shared<Container<ElementType> >(LocationHost, *(rawData->getData<ElementType>()));
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
		auto imageData = dynamic_pointer_cast<const USImage>(_imageData);
		if (imageData)
		{
			//std::cout << "output image data type: " << imageData->getDataType() << std::endl;
			switch (imageData->getDataType())
			{
			case TypeUint8:
				successframeNum = addImageTemplated<uint8_t>(imageData);
				break;
			case TypeInt16:
				successframeNum = addImageTemplated<int16_t>(imageData);
				break;
			case TypeFloat:
				successframeNum = addImageTemplated<float>(imageData);
				break;
			default:
				logging::log_error("MetaImageOutputDevice: Image element type not supported");
				break;
			}			
		}
		return successframeNum;
	}

	std::pair<bool, size_t> MetaImageOutputDevice::addUSRawData(shared_ptr<const RecordObject> _rawData)
	{
		std::pair<bool, size_t> successframeNum = std::make_pair(false, 0);
		auto rawData = dynamic_pointer_cast<const USRawData>(_rawData);
		if (rawData)
		{
			switch (rawData->getDataType())
			{
			case TypeUint8:
				successframeNum = addUSRawDataTemplated<uint8_t>(rawData);
				break;
			case TypeInt16:
				successframeNum = addUSRawDataTemplated<int16_t>(rawData);
				break;
			case TypeFloat:
				successframeNum = addUSRawDataTemplated<float>(rawData);
				break;
			default:
				logging::log_error("MetaImageOutputDevice: Raw data element type not supported");
				break;
			}
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
