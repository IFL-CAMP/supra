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

#include <exception>
#include <sstream>
#include <memory>
#include <vector>
#include <cassert>

#ifdef HAVE_CUDA
#include "utilities/cudaUtility.h"
#endif

namespace supra
{
	enum ContainerLocation
	{
		LocationHost,
		LocationGpu,
		LocationBoth
	};

	template<typename T>
	class Container
	{
	public:
#ifdef HAVE_CUDA
		typedef cudaStream_t ContainerStreamType;
#else
		typedef int ContainerStreamType;
#endif

		Container(ContainerLocation location, ContainerStreamType associatedStream, size_t numel)
		{
#ifndef HAVE_CUDA
			location = LocationHost;
#endif
			m_numel = numel;
			m_location = location;
			m_associatedStream = associatedStream;

			m_buffer = nullptr;
			switch (m_location)
			{
			case LocationGpu:
#ifdef HAVE_CUDA
				cudaSafeCall(cudaMalloc((void**)&m_buffer, m_numel * sizeof(T)));
#endif
				break;
			case LocationBoth:
#ifdef HAVE_CUDA
				cudaSafeCall(cudaMallocManaged((void**)&m_buffer, m_numel * sizeof(T)));
#endif
				break;
			case LocationHost:
#ifdef HAVE_CUDA
				cudaSafeCall(cudaMallocHost((void**)&m_buffer, m_numel * sizeof(T)));
#else
				m_buffer = new T[m_numel];
#endif
				break;
			default:
				throw std::runtime_error("invalid argument: Container: Unknown location given");
			}
			if (!m_buffer)
			{
				std::stringstream s;
				s << "bad alloc: Container: Error allocating buffer of size " << m_numel << " in "
					<< (m_location == LocationHost ? "LocationHost" : (m_location == LocationGpu ? "LocationGpu" : "LocationBoth"));
				throw std::runtime_error(s.str());
			}
		};
		Container(ContainerLocation location, ContainerStreamType associatedStream, const std::vector<T> & data)
			:Container(location, associatedStream, data.size())
		{
#ifdef HAVE_CUDA
			if(location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), data.data(), this->size() * sizeof(T), cudaMemcpyHostToDevice, associatedStream));
			}
			else if(location == LocationBoth)
			{
				cudaSafeCall(cudaMemcpy(this->get(), data.data(), this->size() * sizeof(T), cudaMemcpyDefault));
			}
			else
			{
				std::copy(data.begin(), data.end(), this->get());
			}
#else
			std::copy(data.begin(), data.end(), this->get());
#endif
		};
		Container(ContainerLocation location, ContainerStreamType associatedStream, const T* dataBegin, const T* dataEnd)
			:Container(location, associatedStream, dataEnd - dataBegin)
		{
#ifdef HAVE_CUDA
			cudaSafeCall(cudaMemcpy(this->get(), dataBegin, this->size() * sizeof(T), cudaMemcpyDefault));
#else
			std::copy(data.begin(), data.end(), this->get());
#endif
		};
		Container(ContainerLocation location, const Container<T>& source)
			: Container(location, source.getStream(), source.size())
		{
			if (source.m_location == LocationHost && location == LocationHost)
			{
				std::copy(source.get(), source.get() + source.size(), this->get());
			}
			else if (source.m_location == LocationHost && location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), source.get(), source.size() * sizeof(T), cudaMemcpyHostToDevice, source.getStream()));
			}
			else if (source.m_location == LocationGpu && location == LocationHost)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), source.get(), source.size() * sizeof(T), cudaMemcpyDeviceToHost, source.getStream()));
				cudaSafeCall(cudaStreamSynchronize(source.getStream()));
			}
			else if (source.m_location == LocationGpu && location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpy(this->get(), source.get(), source.size() * sizeof(T), cudaMemcpyDefault));
			}
			else
			{
				cudaSafeCall(cudaMemcpy(this->get(), source.get(), source.size() * sizeof(T), cudaMemcpyDefault));
			}
		};
		~Container()
		{
			switch (m_location)
			{
			case LocationGpu:
#ifdef HAVE_CUDA
				cudaFree(m_buffer);
#endif
				break;
			case LocationBoth:
#ifdef HAVE_CUDA
				cudaFree(m_buffer);
#endif
				break;
			case LocationHost:
#ifdef HAVE_CUDA
				cudaFreeHost(m_buffer);
#else
				delete[] m_buffer;
#endif
				break;
			default:
				break;
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
			cudaSafeCall(cudaMemcpy(dst, this->get(), this->size() * sizeof(T), cudaMemcpyDefault));
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

		/*void associateStream(ContainerStreamType stream)
		{
			m_associatedStream = stream;
		}*/
	private:
		// The number of elements this container can store
		size_t m_numel;
		ContainerLocation m_location;

		ContainerStreamType m_associatedStream;

		T* m_buffer;
	};
}

#endif //!__CONTAINER_H__
