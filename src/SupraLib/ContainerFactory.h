// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2017, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __CONTAINERFACTORY_H__
#define __CONTAINERFACTORY_H__

#ifdef HAVE_CUDA
#include "utilities/cudaUtility.h"
#endif

#include <vector>
#include <mutex>
#include <thread>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_queue.h>

namespace supra
{
	enum ContainerLocation
	{
		LocationHost,
		LocationGpu,
		LocationBoth,
		LocationINVALID
	};

	class ContainerFactory
	{
	public:
#ifdef HAVE_CUDA
		typedef cudaStream_t ContainerStreamType;
#else
		typedef int ContainerStreamType;
#endif
		
		static ContainerStreamType getNextStream();

	protected:
		static uint8_t* acquireMemory(size_t numBytes, ContainerLocation location);
		static void returnMemory(uint8_t* pointer, size_t numBytes, ContainerLocation location);

	private:
		static void initStreams();
	
		static constexpr size_t sm_numberStreams = 8;

		static std::vector<ContainerStreamType> sm_streams;
		static size_t sm_streamIndex;
		static std::mutex sm_streamMutex;

		static constexpr double sm_deallocationTimeout = 60; // [seconds]

		static uint8_t* allocateMemory(size_t numBytes, ContainerLocation location);
		static void freeBuffers(size_t numBytesMin, ContainerLocation location);
		static void freeOldBuffers();
		static void garbageCollectionThreadFunction();
		static void freeMemory(uint8_t* pointer, size_t numBytes, ContainerLocation location);

		static std::array<tbb::concurrent_unordered_map<size_t, tbb::concurrent_queue<std::pair<uint8_t*, double> > >, LocationINVALID> sm_bufferMaps;
		static std::thread sm_garbageCollectionThread;
	};

	class ContainerFactoryContainerInterface : public ContainerFactory
	{
		template <typename T> friend class Container;
	};
}

#endif //!__CONTAINERFACTORY_H__
