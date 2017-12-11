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

#ifndef __MHDSEQUENCEWRITER_H__
#define __MHDSEQUENCEWRITER_H__

#ifdef HAVE_DEVICE_METAIMAGE_OUTPUT

#include <array>
#include <functional>

namespace supra
{
	class MhdSequenceWriter
	{
	public:
		MhdSequenceWriter();
		~MhdSequenceWriter();

		void open(std::string basefilename);

		bool isOpen();

		template <typename ValueType>
		size_t addImage(const ValueType* imageData, size_t w, size_t h, size_t d,
			double timestamp, double spacing, std::function<void(void)> deleteCallback = std::function<void(void)>());

		void addTracking(size_t frameNumber, std::array<double, 16> T, bool transformValid, std::string transformName);

		void close();
	private:
		bool m_wroteHeaders;
		size_t m_nextFrameNumber;
		std::ofstream m_mhdFile;
		std::ofstream m_rawFile;
		std::string m_baseFilename;
		std::string m_mhdFilename;
		std::string m_rawFilename;
		std::string m_rawFilenameNoPath;

		std::streampos m_positionImageCount;
	};
}

#endif //!HAVE_DEVICE_METAIMAGE_OUTPUT

#endif //!__MHDSEQUENCEWRITER_H__
