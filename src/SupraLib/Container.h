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

#ifndef __CONTAINER_H__
#define __CONTAINER_H__

#include "ContainerFactory.h"
#ifdef HAVE_CUDA
#include "utilities/cudaUtility.h"
#endif

#include <exception>
#include <memory>
#include <vector>
#include <cassert>

namespace supra
{
	template<typename T>
	class Container
	{
	public:
		typedef ContainerFactory::ContainerStreamType ContainerStreamType;

		Container(ContainerLocation location, ContainerStreamType associatedStream, size_t numel)
		{
#ifndef HAVE_CUDA
			location = LocationHost;
#endif
#ifdef HAVE_CUDA
			m_creationEvent = nullptr;
#endif
			m_numel = numel;
			m_location = location;
			m_associatedStream = associatedStream;

			m_buffer = reinterpret_cast<T*>(ContainerFactoryContainerInterface::acquireMemory(
				m_numel * sizeof(T), m_location));
		};
		Container(ContainerLocation location, ContainerStreamType associatedStream, const std::vector<T> & data, bool waitFinished = true)
			:Container(location, associatedStream, data.size())
		{
#ifdef HAVE_CUDA
			if(location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), data.data(), this->size() * sizeof(T), cudaMemcpyHostToDevice, associatedStream));
				createAndRecordEvent();
			}
			else if(location == LocationBoth)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), data.data(), this->size() * sizeof(T), cudaMemcpyHostToDevice, associatedStream));
				createAndRecordEvent();
			}
			else
			{
				std::copy(data.begin(), data.end(), this->get());
			}
			if (waitFinished)
			{
				waitCreationFinished();
			}
#else
			std::copy(data.begin(), data.end(), this->get());
#endif
		};
		Container(ContainerLocation location, ContainerStreamType associatedStream, const T* dataBegin, const T* dataEnd, bool waitFinished = true)
			:Container(location, associatedStream, dataEnd - dataBegin)
		{
#ifdef HAVE_CUDA
			if (location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), dataBegin, this->size() * sizeof(T), cudaMemcpyHostToDevice, associatedStream));
				createAndRecordEvent();
			}
			else if (location == LocationBoth)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), dataBegin, this->size() * sizeof(T), cudaMemcpyHostToDevice, associatedStream));
				createAndRecordEvent();
			}
			else
			{
				std::copy(dataBegin, dataBegin + this->size() * sizeof(T), this->get());
			}
			if (waitFinished)
			{
				waitCreationFinished();
			}
#else
			std::copy(dataBegin, dataBegin + this->size() * sizeof(T), this->get());
#endif
		};
		Container(ContainerLocation location, const Container<T>& source, bool waitFinished = true)
			: Container(location, source.getStream(), source.size())
		{
			if (source.m_location == LocationHost && location == LocationHost)
			{
				std::copy(source.get(), source.get() + source.size(), this->get());
			}
			else if (source.m_location == LocationHost && location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), source.get(), source.size() * sizeof(T), cudaMemcpyHostToDevice, source.getStream()));
				createAndRecordEvent();
			}
			else if (source.m_location == LocationGpu && location == LocationHost)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), source.get(), source.size() * sizeof(T), cudaMemcpyDeviceToHost, source.getStream()));
				createAndRecordEvent();
			}
			else if (source.m_location == LocationGpu && location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), source.get(), source.size() * sizeof(T), cudaMemcpyDeviceToDevice, source.getStream()));
				createAndRecordEvent();
			}
			else
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), source.get(), source.size() * sizeof(T), cudaMemcpyDefault, source.getStream()));
				createAndRecordEvent();
			}
			if (waitFinished)
			{
				waitCreationFinished();
			}
		};
		~Container()
		{
			auto ret = cudaStreamQuery(m_associatedStream);
			if (ret != cudaSuccess && ret != cudaErrorNotReady && ret != cudaErrorCudartUnloading)
			{
				cudaSafeCall(ret);
			}
			// If the driver is currently unloading, we cannot free the memory in any way. Exit will clean up.
			else if(ret != cudaErrorCudartUnloading)
			{
				if (ret == cudaSuccess)
				{
					ContainerFactoryContainerInterface::returnMemory(reinterpret_cast<uint8_t*>(m_buffer), m_numel * sizeof(T), m_location);
				}
				else
				{
					auto buffer = m_buffer;
					auto numel = m_numel;
					auto location = m_location;
					addCallbackStream([buffer, numel, location](cudaStream_t s, cudaError_t e) -> void {
						ContainerFactoryContainerInterface::returnMemory(reinterpret_cast<uint8_t*>(buffer), numel * sizeof(T), location);
					});
				}
			}
		};

		const T* get() const { return m_buffer; };
		T* get() { return m_buffer; };

		T* getCopyHostRaw() const
		{
#ifdef HAVE_CUDA
			auto ret = new T[this->size()];
			
			if(m_location == LocationHost)
			{
				std::copy(this->get(), this->get() + this->size(), ret);
			}
			else if(m_location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(ret, this->get(), this->size() * sizeof(T), cudaMemcpyDeviceToHost, getStream()));
				cudaSafeCall(cudaStreamSynchronize(getStream()));				
			}
			else 
			{
				cudaSafeCall(cudaMemcpy(ret, this->get(), this->size() * sizeof(T), cudaMemcpyDefault));
			}
			return ret;
#else
			return nullptr;
#endif
		}

		void copyTo(T* dst, size_t maxSize) const
		{
#ifdef HAVE_CUDA
			assert(maxSize >= this->size());
			if (m_location == LocationHost)
			{
				std::copy(this->get(), this->get() + this->size(), dst);
			}
			else if (m_location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(dst, this->get(), this->size() * sizeof(T), cudaMemcpyDeviceToHost, getStream()));
				cudaSafeCall(cudaStreamSynchronize(getStream()));
			}
			else
			{
				cudaSafeCall(cudaMemcpy(dst, this->get(), this->size() * sizeof(T), cudaMemcpyDefault));
			}
#endif
		}

		void waitCreationFinished()
		{
#ifdef HAVE_CUDA
			if (m_creationEvent)
			{
				cudaSafeCall(cudaEventSynchronize(m_creationEvent));
				cudaSafeCall(cudaEventDestroy(m_creationEvent));
				m_creationEvent = nullptr;
			}
#endif
		}

		// returns the number of elements that can be stored in this container
		size_t size() const { return m_numel; };

		bool isHost() const { return m_location == ContainerLocation::LocationHost; };
		bool isGPU() const { return m_location == ContainerLocation::LocationGpu; };
		bool isBoth() const { return m_location == ContainerLocation::LocationBoth; };
		ContainerLocation getLocation() const { return m_location; };
		ContainerStreamType getStream() const
		{
			return m_associatedStream;
		}
	private:
		void createAndRecordEvent()
		{
			if (!m_creationEvent)
			{
				//cudaSafeCall(cudaEventCreateWithFlags(&m_creationEvent, cudaEventBlockingSync | cudaEventDisableTiming));
				cudaSafeCall(cudaEventCreateWithFlags(&m_creationEvent, cudaEventDisableTiming));
			}
			cudaSafeCall(cudaEventRecord(m_creationEvent, m_associatedStream));
		}

		void addCallbackStream(std::function<void(cudaStream_t, cudaError_t)> func)
		{
			auto funcPointer = new std::function<void(cudaStream_t, cudaError_t)>(func);
			cudaSafeCall(cudaStreamAddCallback(m_associatedStream, &(Container<T>::cudaDeleteCallback), funcPointer, 0));
		}

		static void CUDART_CB cudaDeleteCallback(cudaStream_t stream, cudaError_t status, void* userData)
		{
			std::unique_ptr<std::function<void(cudaStream_t, cudaError_t)> > func =
				std::unique_ptr<std::function<void(cudaStream_t, cudaError_t)> >(
					reinterpret_cast<std::function<void(cudaStream_t, cudaError_t)>*>(userData));
			(*func)(stream, status);
		}

		// The number of elements this container can store
		size_t m_numel;
		ContainerLocation m_location;

		ContainerStreamType m_associatedStream;
		T* m_buffer;

#ifdef HAVE_CUDA
		cudaEvent_t m_creationEvent;
#endif
	};
}

#endif //!__CONTAINER_H__
