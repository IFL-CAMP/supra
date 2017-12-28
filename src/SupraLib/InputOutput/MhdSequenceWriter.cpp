// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016-2017, all rights reserved,
//		Rüdiger Göbl
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
//
// ================================================================================================

#include "MhdSequenceWriter.h"
#include "utilities/Logging.h"

#include <sstream>
#include <iomanip>

namespace supra
{
	MhdSequenceWriter::MhdSequenceWriter()
		: m_wroteHeaders(false)
		, m_nextFrameNumber(0)
		, m_memoryBufferSize(0)
		, m_closing(false)
	{};

	MhdSequenceWriter::~MhdSequenceWriter()
	{
		closeFiles();
	};

	void MhdSequenceWriter::open(std::string basefilename, size_t memoryBufferSize)
	{
		m_baseFilename = basefilename;
		m_mhdFilename = m_baseFilename + ".mhd";
		m_mhdFile.open(m_mhdFilename, std::ios_base::out | std::ios_base::trunc);

		m_rawFilename = m_baseFilename + ".raw";
		m_rawFilenameNoPath = m_baseFilename.substr(m_baseFilename.find_last_of("/\\") + 1) + ".raw";
		// Turn of buffering for raw writer to increase throughput
		m_rawFile.rdbuf()->pubsetbuf(0, 0);
		m_rawFile.open(m_rawFilename, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

		m_mhdFile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

		m_memoryBufferSize = memoryBufferSize;
		m_writerThread = std::thread(&MhdSequenceWriter::writerThread, this);
	}

	bool MhdSequenceWriter::isOpen()
	{
		return !m_closing && m_mhdFile.is_open() && m_rawFile.is_open();
	}

	template <typename ValueType>
	std::pair<bool, size_t> MhdSequenceWriter::addImage(const ValueType* imageData, size_t w, size_t h, size_t d, size_t channels,
		double timestamp, double spacing, std::function<void(const uint8_t*, size_t)> deleteCallback)
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
					m_mhdFile << "ElementNumberOfChannels = " << channels << "\n";
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

				//add data to the write queue
				bool written = addImageQueue(reinterpret_cast<const uint8_t*>(imageData), w*h*d*channels * sizeof(ValueType), deleteCallback);

				if(!written)
				{ 
					return std::make_pair(false, 0);
				}
				else
				{
					size_t thisFrameNum = m_nextFrameNumber;
					m_nextFrameNumber++;

					//write the image sequence info
					std::stringstream frameNumStr;
					frameNumStr << std::setfill('0') << std::setw(4) << thisFrameNum;
					m_mhdFile << "Seq_Frame" << frameNumStr.str() << "_ImageStatus = OK\n"
						<< "Seq_Frame" << frameNumStr.str() << "_Timestamp = " << timestamp << "\n";

					//update the sequence size
					m_mhdFile.seekp(m_positionImageCount);
					m_mhdFile << thisFrameNum + 1;
					m_mhdFile.seekp(0, std::ios_base::end);

					return std::make_pair(true, thisFrameNum);
				}
			}
			else {
				logging::log_warn("Could not write frame to MHD, file already contains 9999 frames.");
				return std::make_pair(false, 0);
			}
		}
		else {
			logging::log_error("Could not write frame to MHD, sizes are inconsistent. (w = ", w, ", h = ", h, ", d = ", d, ", channels = ", channels, ")");
			return std::make_pair(false, 0);
		}
	}

	template
		std::pair<bool, size_t> MhdSequenceWriter::addImage<uint8_t>(const uint8_t* imageData, size_t w, size_t h, size_t d, size_t channels,
		double timestamp, double spacing, std::function<void(const uint8_t*, size_t)> deleteCallback);
	template
		std::pair<bool, size_t>MhdSequenceWriter::addImage<int16_t>(const int16_t* imageData, size_t w, size_t h, size_t d, size_t channels,
		double timestamp, double spacing, std::function<void(const uint8_t*, size_t)> deleteCallback);

	void MhdSequenceWriter::addTracking(size_t frameNumber, std::array<double, 16> T, bool transformValid, std::string transformName)
	{
		std::stringstream frameNumStr;
		frameNumStr << std::setfill('0') << std::setw(4) << frameNumber;
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

	void MhdSequenceWriter::close()
	{
		m_closing = true;
		m_writerThread.detach();
	}

	void MhdSequenceWriter::closeFiles()
	{
		if (m_mhdFile.is_open())
		{
			m_mhdFile << "ElementDataFile = " << m_rawFilenameNoPath << "\n";
			m_mhdFile.close();
		}
		{
			std::unique_lock<std::mutex> l(m_rawFileMutex);
			if (m_rawFile.is_open())
			{
				m_rawFile.close();
			}
		}
	}

	bool MhdSequenceWriter::addImageQueue(const uint8_t * imageData, size_t numel, std::function<void(const uint8_t*, size_t)> deleteCallback)
	{
		std::unique_lock<std::mutex> l(m_queueMutex);

		if (m_writeQueue.size() * numel <= m_memoryBufferSize)
		{
			m_writeQueue.push(std::make_tuple(imageData, numel, deleteCallback));
			m_queueConditionVariable.notify_one();

			return true;
		}
		return false;
	}

	void MhdSequenceWriter::writerThread()
	{
		while (!m_closing)
		{
			std::tuple<const uint8_t*, size_t, std::function<void(const uint8_t*, size_t)> > queueEntry;
			bool haveEntry = false;
			{
				std::unique_lock<std::mutex> l(m_queueMutex);
				if (m_writeQueue.size() > 0)
				{
					queueEntry = m_writeQueue.front();
					m_writeQueue.pop();
					haveEntry = true;
				}
			}
			if (haveEntry)
			{
				addImageInternal(std::get<0>(queueEntry), std::get<1>(queueEntry), std::get<2>(queueEntry));
			}
			{
				std::unique_lock<std::mutex> l(m_queueMutex);
				if (m_writeQueue.size() == 0)
				{
					m_queueConditionVariable.wait(l);
				}
			}
		}

		//After the sequence is marked to be closed, no other thread will interfere anymore
		{
			std::unique_lock<std::mutex> l(m_queueMutex);
			while (m_writeQueue.size() > 0)
			{
				auto queueEntry = m_writeQueue.front();
				m_writeQueue.pop();
				addImageInternal(std::get<0>(queueEntry), std::get<1>(queueEntry), std::get<2>(queueEntry));
			}
		}

		// Now that everything has been written, this object can be destroyed
		delete this;
	}

	void MhdSequenceWriter::addImageInternal(const uint8_t * imageData, size_t numel, std::function<void(const uint8_t*, size_t)> deleteCallback)
	{
		{
			std::unique_lock<std::mutex> l(m_rawFileMutex);
			m_rawFile.write(reinterpret_cast<const char*>(imageData), numel);
		}
		deleteCallback(imageData, numel);
	}
}