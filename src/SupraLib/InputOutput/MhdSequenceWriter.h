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
#include <fstream>
#include <string>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <thread>

namespace supra
{
#ifdef _WIN32
	#define MHDSEQUENCEWRITER_MEMORY_BUFFER_DEFAULT_SIZE (1*1024*1024*(size_t)(1024)) // [Bytes]
#else
	#define MHDSEQUENCEWRITER_MEMORY_BUFFER_DEFAULT_SIZE (4*1024*1024*(size_t)(1024)) // [Bytes]
#endif

	class MhdSequenceWriter
	{
	public:
		MhdSequenceWriter();
		
		void open(std::string basefilename, size_t memoryBufferSize = MHDSEQUENCEWRITER_MEMORY_BUFFER_DEFAULT_SIZE);

		bool isOpen();

		template <typename ValueType>
		std::pair<bool, size_t> addImage(const ValueType* imageData, size_t w, size_t h, size_t d,
			double timestamp, double spacing,
			std::function<void(const uint8_t*, size_t)> deleteCallback = std::function<void(const uint8_t*, size_t)>());

		void addTracking(size_t frameNumber, std::array<double, 16> T, bool transformValid, std::string transformName);
		void close();
	private:
		~MhdSequenceWriter();

		void closeFiles();

		bool addImageQueue(const uint8_t* imageData, size_t numel, std::function<void(const uint8_t*, size_t)> deleteCallback);
		void writerThread();
		void addImageInternal(const uint8_t* imageData, size_t numel, std::function<void(const uint8_t*, size_t)> deleteCallback);

		bool m_wroteHeaders;
		size_t m_nextFrameNumber;
		size_t m_memoryBufferSize;
		std::ofstream m_mhdFile;
		std::ofstream m_rawFile;
		std::string m_baseFilename;
		std::string m_mhdFilename;
		std::string m_rawFilename;
		std::string m_rawFilenameNoPath;

		std::streampos m_positionImageCount;

		std::queue<std::tuple<const uint8_t*, size_t, std::function<void(const uint8_t*, size_t)> > > m_writeQueue;

		std::atomic<bool> m_closing;
		std::mutex m_rawFileMutex;
		std::mutex m_queueMutex;
		std::condition_variable m_queueConditionVariable;
		std::thread m_writerThread;
	};
}

#endif //!HAVE_DEVICE_METAIMAGE_OUTPUT

#endif //!__MHDSEQUENCEWRITER_H__
