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

#include "ContainerFactory.h"

#include <utilities/Logging.h>
#include <utilities/utility.h>

#include <sstream>
#include <cassert>
using namespace std;

namespace supra
{
	ContainerFactory::ContainerStreamType ContainerFactory::getNextStream()
	{
		std::lock_guard<std::mutex> streamLock(sm_streamMutex);

		if (sm_streams.size() == 0)
		{
			initStreams();
		}

		size_t streamIndex = sm_streamIndex;
		sm_streamIndex = (sm_streamIndex + 1) % sm_numberStreams;
		return sm_streams[streamIndex];
	}
	uint8_t* ContainerFactory::acquireMemory(size_t numBytes, ContainerLocation location)
	{
		assert(location < LocationINVALID);
		
		// Check whether the queue for this location and size has a buffer left
		uint8_t* buffer = nullptr;
		{
			// by directly accessing the desired length in the map sm_bufferMaps[location],
			// the map entry is created if it does not already exist. That means the map is
			// modified here
			tbb::concurrent_queue<std::pair<uint8_t*, double> >* queuePointer =
				&(sm_bufferMaps[location][numBytes]);

			std::pair<uint8_t*, double> queueEntry;
			if (queuePointer->try_pop(queueEntry))
			{
				// If yes, just return this already allocated buffer
				buffer = queueEntry.first;
			}
		}

		// If the queue did not contain a buffer, allocate a new one
		if (!buffer)
		{
			// Check whether there is enough free space for the requested buffer. 
			size_t memoryFree;
			size_t memoryTotal;
#ifdef HAVE_CUDA
			if (location == LocationGpu || location == LocationBoth)
			{
				cudaSafeCall(cudaMemGetInfo(&memoryFree, &memoryTotal));
				memoryFree = static_cast<size_t>(std::max(static_cast<double>(memoryFree) - (static_cast<double>(memoryTotal) *0.02), 0.0));
			}
			else
#endif
			{
				// For the host memory we just rely on the 
				memoryFree = numBytes;
			}

			// If not, relase enough unused buffers, starting with the ones that have been returned the longest time ago.
			if (memoryFree < numBytes)
			{
				freeBuffers(numBytes, location);
			}

			// additionaly, release memory that has been returned over XX (e.g. 30) seconds ago
			freeOldBuffers();

			// Now that we have made the required memory available, we can allocate the buffer
			buffer = allocateMemory(numBytes, location);
		}

		return buffer;
	}

	void ContainerFactory::returnMemory(uint8_t* pointer, size_t numBytes, ContainerLocation location)
	{
		assert(location < LocationINVALID);

		// do not free here, just put it back to the queues with the time it was returned at
		double returnTime = getCurrentTime();

		// Put buffer back to queue
		{
			tbb::concurrent_queue<std::pair<uint8_t*, double> >* queuePointer =
				&(sm_bufferMaps[location][numBytes]);

			queuePointer->push(std::make_pair(pointer, returnTime));
		}
	}
	void ContainerFactory::initStreams()
	{
		logging::log_log("ContainerFactory: Initializing ", sm_numberStreams, " streams.");
		sm_streamIndex = 0;
#ifdef HAVE_CUDA
		sm_streams.resize(sm_numberStreams);
		for (size_t k = 0; k < sm_numberStreams; k++)
		{
			cudaSafeCall(cudaStreamCreateWithFlags(&(sm_streams[k]), cudaStreamNonBlocking));
		}
#else
		sm_streams.resize(sm_numberStreams, 0);
#endif
	}

	uint8_t * ContainerFactory::allocateMemory(size_t numBytes, ContainerLocation location)
	{
		uint8_t* buffer = nullptr;
		switch (location)
		{
		case LocationGpu:
#ifdef HAVE_CUDA
			cudaSafeCall(cudaMalloc((void**)&buffer, numBytes));
#endif
			break;
		case LocationBoth:
#ifdef HAVE_CUDA
			cudaSafeCall(cudaMallocManaged((void**)&buffer, numBytes));
#endif
			break;
		case LocationHost:
#ifdef HAVE_CUDA
			cudaSafeCall(cudaMallocHost((void**)&buffer, numBytes));
#else
			buffer = new uint8_t[numBytes];
#endif
			break;
		default:
			throw std::runtime_error("invalid argument: Container: Unknown location given");
		}
		if (!buffer)
		{
			std::stringstream s;
			s << "bad alloc: Container: Error allocating buffer of size " << numBytes << " in "
				<< (location == LocationHost ? "LocationHost" : (location == LocationGpu ? "LocationGpu" : "LocationBoth"));
			throw std::runtime_error(s.str());
		}

		return buffer;
	}

	void ContainerFactory::freeBuffers(size_t numBytesMin, ContainerLocation location)
	{
		size_t numBytesFreed = 0;
		size_t numBuffersFreed;
		do 
		{
			numBuffersFreed = 0;
			// by traversing the map sm_bufferMaps[location] we never create new entries, but only modifiy those already present.
			for (auto mapIterator = sm_bufferMaps[location].begin(); mapIterator != sm_bufferMaps[location].end(); mapIterator++)
			{
				size_t numBytesBuffer = mapIterator->first;
				std::pair<uint8_t*, double> queueEntry;
				if(mapIterator->second.try_pop(queueEntry))
				{
					// If there is an element in this queue, remove it and free the memory
					freeMemory(queueEntry.first, numBytesBuffer, location);
					numBytesFreed += numBytesBuffer;
					numBuffersFreed++;
				}
			}
		} while (numBytesFreed < numBytesMin && numBuffersFreed > 0);
	}

	void ContainerFactory::freeOldBuffers()
	{
		double currentTime = getCurrentTime();
		double deleteTime = currentTime - sm_deallocationTimeout;
		for (ContainerLocation location = LocationHost; location < LocationINVALID; location = static_cast<ContainerLocation>(location + 1))
		{
			for (auto mapIterator = sm_bufferMaps[location].begin(); mapIterator != sm_bufferMaps[location].end(); mapIterator++)
			{
				size_t numBytesBuffer = mapIterator->first;

				double lastTime = 0.0;
				while(!mapIterator->second.empty() && lastTime < deleteTime)
				{
					// If there is an element in this queue, remove it and free the memory
					std::pair<uint8_t*, double> bufferPair;
					if (mapIterator->second.try_pop(bufferPair))
					{
						lastTime = bufferPair.second;
						if (lastTime < deleteTime)
						{
							freeMemory(bufferPair.first, numBytesBuffer, location);
						}
						else
						{
							// oops, we should not have taken that element from the queue. Let's just put it back.
							// Yes, it will be in the wrong temporal order, but that will be solved in a while on its own
							mapIterator->second.push(bufferPair);
						}
					}
				}
			}
		}
	}

	void ContainerFactory::garbageCollectionThreadFunction()
	{
		sm_garbageCollectionThread.detach();
		while (true)
		{
			ContainerFactory::freeOldBuffers();
			std::this_thread::sleep_for(std::chrono::duration<double>(sm_deallocationTimeout));
		}
	}

	void ContainerFactory::freeMemory(uint8_t * pointer, size_t numBytes, ContainerLocation location)
	{
		switch (location)
		{
		case LocationGpu:
#ifdef HAVE_CUDA
			cudaFree(pointer);
#endif
			break;
		case LocationBoth:
#ifdef HAVE_CUDA
			cudaFree(pointer);
#endif
			break;
		case LocationHost:
#ifdef HAVE_CUDA
			cudaFreeHost(pointer);
#else
			delete[] pointer;
#endif
			break;
		default:
			break;
		}
	}

	std::vector<ContainerFactory::ContainerStreamType> ContainerFactory::sm_streams = {};
	size_t ContainerFactory::sm_streamIndex = 0;
	std::mutex ContainerFactory::sm_streamMutex;

	constexpr double ContainerFactory::sm_deallocationTimeout;

	std::array<tbb::concurrent_unordered_map<size_t, tbb::concurrent_queue<std::pair<uint8_t*, double> > >, LocationINVALID> ContainerFactory::sm_bufferMaps;

	std::thread ContainerFactory::sm_garbageCollectionThread(&ContainerFactory::garbageCollectionThreadFunction);
}
