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

#include <fstream>
#include <sstream>
#include <iomanip>

namespace supra
{
	MhdSequenceWriter::MhdSequenceWriter()
		: m_wroteHeaders(false)
		, m_nextFrameNumber(0)
	{};

	MhdSequenceWriter::~MhdSequenceWriter()
	{
		if (m_mhdFile.is_open() || m_rawFile.is_open())
		{
			close();
		}
	};

	void MhdSequenceWriter::open(std::string basefilename)
	{
		m_baseFilename = basefilename;
		m_mhdFilename = m_baseFilename + ".mhd";
		m_mhdFile.open(m_mhdFilename, std::ios_base::out | std::ios_base::trunc);

		m_rawFilename = m_baseFilename + ".raw";
		m_rawFilenameNoPath = m_baseFilename.substr(m_baseFilename.find_last_of("/\\") + 1) + ".raw";
		m_rawFile.open(m_rawFilename, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

		//TODO: If performace is not sufficient, try unbuffered output (must happen before opening the file)
		//m_mhdFile.rdbuf()->pubsetbuf(0, 0);
		//m_rawFile.rdbuf()->pubsetbuf(0, 0);

		m_mhdFile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
	}

	bool MhdSequenceWriter::isOpen()
	{
		return m_mhdFile.is_open() && m_rawFile.is_open();
	}

	template <typename ValueType>
	size_t MhdSequenceWriter::addImage(const ValueType* imageData, size_t w, size_t h, size_t d, 
		double timestamp, double spacing, std::function<void(void)> deleteCallback)
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

	template <>
	size_t MhdSequenceWriter::addImage<uint8_t>(const uint8_t* imageData, size_t w, size_t h, size_t d,
		double timestamp, double spacing, std::function<void(void)> deleteCallback);
	template <>
	size_t MhdSequenceWriter::addImage<int16_t>(const int16_t* imageData, size_t w, size_t h, size_t d,
		double timestamp, double spacing, std::function<void(void)> deleteCallback);

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
}