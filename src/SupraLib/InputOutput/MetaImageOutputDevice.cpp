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

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <cassert>

#include "MetaImageOutputDevice.h"

#include <SyncRecordObject.h>
#include <USImage.h>
#include <Beamformer/USRawData.h>
#include <TrackerDataSet.h>
#include <utilities/Logging.h>

using namespace std;
namespace supra
{
	using namespace logging;

	class MhdSequenceWriter
	{
	public:
		MhdSequenceWriter()
			: m_wroteHeaders(false)
			, m_nextFrameNumber(0)
		{};
		~MhdSequenceWriter()
		{
			if (m_mhdFile.is_open() || m_rawFile.is_open())
			{
				close();
			}
		};

		void open(string basefilename)
		{
			m_baseFilename = basefilename;
			m_mhdFilename = m_baseFilename + ".mhd";
			m_mhdFile.open(m_mhdFilename, ios_base::out | ios_base::trunc);

			m_rawFilename = m_baseFilename + ".raw";
			m_rawFilenameNoPath = m_baseFilename.substr(m_baseFilename.find_last_of("/\\") + 1) + ".raw";
			m_rawFile.open(m_rawFilename, ios_base::out | ios_base::trunc | ios_base::binary);

			//TODO: If performace is not sufficient, try unbuffered output (must happen before opening the file)
			//m_mhdFile.rdbuf()->pubsetbuf(0, 0);
			//m_rawFile.rdbuf()->pubsetbuf(0, 0);

			m_mhdFile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
		}

		bool isOpen()
		{
			return m_mhdFile.is_open() && m_rawFile.is_open();
		}

		template <typename ValueType>
		size_t addImage(const ValueType* imageData, size_t w, size_t h, size_t d, double timestamp, double spacing)
		{
			static_assert(
				std::is_same<ValueType, uint8_t>::value ||
				std::is_same<ValueType, int16_t>::value,
				"MHD only implemented for uchar and short at the moment");
			if (w > 0 && h > 0 && d > 0)
			{
				if (m_nextFrameNumber < 10000)
				{
					if (!m_wroteHeaders)
					{
						//this is the first image in the sequence: write the headers
						m_mhdFile << "ObjectType = Image\n"
							<< (d == 1 ? "NDims = 3\n" : "NDims = 4\n")
							<< "DimSize = " << w << " " << h << " ";
						if (d > 1)
						{
							m_mhdFile << d << " ";
						}
						//remember where the 3rd dimension = number of images has to be written to
						m_positionImageCount = m_mhdFile.tellp();
						//place a few spaces as placeholders, assuming trailing whitespace does not hurt
						m_mhdFile << "          \n"; // 10 spaces that can be replaced with numbers
						m_mhdFile << "ElementNumberOfChannels = 1\n";
						if (std::is_same<ValueType, uint8_t>::value)
						{
							m_mhdFile << "ElementType = MET_UCHAR\n";
						}
						else if (std::is_same<ValueType, int16_t>::value)
						{
							m_mhdFile << "ElementType = MET_SHORT\n";
						}
						m_mhdFile << "ElementSpacing = " << spacing << " " << spacing << " ";
						if (d > 1)
						{
							m_mhdFile << spacing << " ";
						}
						m_mhdFile << "1\n"
							<< "AnatomicalOrientation = RAI\n"
							<< "BinaryData = True\n"
							<< "BinaryDataByteOrderMSB = False\n"
							<< "CenterOfRotation = 0 0 0\n"
							<< "CompressedData = False\n"
							<< "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
							<< "UltrasoundImageOrientation = MFA\n"
							<< "UltrasoundImageType = BRIGHTNESS\n"
							<< "ElementByteOrderMSB = False\n";
						m_wroteHeaders = true;
					}

					size_t thisFrameNum = m_nextFrameNumber;
					m_nextFrameNumber++;

					//write the data
					m_rawFile.write(reinterpret_cast<const char*>(imageData), w*h*d * sizeof(ValueType));

					//write the image sequence info
					stringstream frameNumStr;
					frameNumStr << setfill('0') << setw(4) << thisFrameNum;
					m_mhdFile << "Seq_Frame" << frameNumStr.str() << "_ImageStatus = OK\n"
						<< "Seq_Frame" << frameNumStr.str() << "_Timestamp = " << timestamp << "\n";

					//update the sequence size
					m_mhdFile.seekp(m_positionImageCount);
					m_mhdFile << thisFrameNum + 1;
					m_mhdFile.seekp(0, ios_base::end);

					return thisFrameNum;
				}
				else {
					log_warn("Could not write frame to MHD, file already contains 9999 frames.");
					return m_nextFrameNumber;
				}
			}
			else {
				log_error("Could not write frame to MHD, sizes are inconsistent. (w = ", w, ", h = ", h, ", d = ", d, ")");
				return m_nextFrameNumber;
			}
		}

		void addTracking(size_t frameNumber, array<double, 16> T, bool transformValid, string transformName)
		{
			stringstream frameNumStr;
			frameNumStr << setfill('0') << setw(4) << frameNumber;
			if (transformValid)
			{
				m_mhdFile << "Seq_Frame" << frameNumStr.str() << "_" << transformName << "ToTrackerTransform =";
				for (int i = 0; i < 16; i++)
				{
					m_mhdFile << " " << T[i];
				}
				m_mhdFile << "\nSeq_Frame" << frameNumStr.str() << "_" << transformName << "ToTrackerTransformStatus = OK\n";
			}
			else
			{
				m_mhdFile << "Seq_Frame" << frameNumStr.str() << "_" << transformName << "ToTrackerTransform = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1";
				m_mhdFile << "\nSeq_Frame" << frameNumStr.str() << "_" << transformName << "ToTrackerTransformStatus = INVALID\n";
			}
		}

		void close()
		{
			if (m_mhdFile.is_open())
			{
				m_mhdFile << "ElementDataFile = " << m_rawFilenameNoPath << "\n";
				m_mhdFile.close();
			}
			if (m_rawFile.is_open())
			{
				m_rawFile.close();
			}
		}
	private:
		bool m_wroteHeaders;
		size_t m_nextFrameNumber;
		ofstream m_mhdFile;
		ofstream m_rawFile;
		string m_baseFilename;
		string m_mhdFilename;
		string m_rawFilename;
		string m_rawFilenameNoPath;

		streampos m_positionImageCount;
	};


	MetaImageOutputDevice::MetaImageOutputDevice(tbb::flow::graph& graph, const std::string & nodeID)
		: AbstractOutput(graph, nodeID)
		, m_filename("output")
		, m_createSequences(false)
		, m_isRecording(false)
		, m_sequencesWritten(0)
		, m_active(true)
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
			m_pWriter->close();

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

		m_pWriter = unique_ptr<MhdSequenceWriter>(new MhdSequenceWriter());
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
				size_t frameNum = addImage(mainRecord);

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

	template <typename ElementType>
	size_t MetaImageOutputDevice::addImageTemplated(shared_ptr<const USImage<ElementType> > imageData)
	{
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
					pDataCopy = imageData->getData()->getCopy(ContainerLocation::LocationHost);
					pData = pDataCopy;
				}

				frameNum = m_pWriter->addImage(pData->get(), imageSize.x, imageSize.y, imageSize.z, imageData->getSyncTimestamp(), resolution);
			}
		}
		return frameNum;
	}

	template <typename ElementType>
	size_t MetaImageOutputDevice::addUSRawDataTemplated(shared_ptr<const USRawData<ElementType> > rawData)
	{
		size_t frameNum = 0;
		if (rawData)
		{
			size_t numChannels = rawData->getNumReceivedChannels();
			size_t numSamples = rawData->getNumSamples();
			size_t numScanlines = rawData->getNumScanlines();

			shared_ptr<Container<int16_t> > pDataCopy;
			shared_ptr<const Container<int16_t> > pData;
			if (rawData->getData()->isHost())
			{
				pData = rawData->getData();
			}
			else {
				pDataCopy = rawData->getData()->getCopy(ContainerLocation::LocationHost);
				pData = pDataCopy;
			}
			double resolution = rawData->getImageProperties()->getSampleDistance();
			frameNum = m_pWriter->addImage(pData->get(), numSamples, numChannels, numScanlines, rawData->getSyncTimestamp(), resolution);
		}
		return frameNum;
	}

	size_t MetaImageOutputDevice::addImage(shared_ptr<const RecordObject> _imageData)
	{
		size_t frameNum = 0;
		auto imageData8Bit = dynamic_pointer_cast<const USImage<uint8_t>>(_imageData);
		if (imageData8Bit)
		{
			frameNum = addImageTemplated<uint8_t>(imageData8Bit);
		}

		auto imageData16Bit = dynamic_pointer_cast<const USImage<int16_t>>(_imageData);
		if (imageData16Bit)
		{
			frameNum = addImageTemplated<int16_t>(imageData16Bit);
		}
		return frameNum;
	}

	size_t MetaImageOutputDevice::addUSRawData(shared_ptr<const RecordObject> _rawData)
	{
		size_t frameNum = 0;
		auto rawData = dynamic_pointer_cast<const USRawData<int16_t>>(_rawData);
		if (rawData)
		{
			frameNum = addUSRawDataTemplated<int16_t>(rawData);
		}
		return frameNum;
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