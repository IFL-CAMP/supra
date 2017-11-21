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
		Container(ContainerLocation location, size_t numel)
		{
#ifndef HAVE_CUDA
			location = LocationHost;
#endif
			m_numel = numel;
			m_location = location;

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
		Container(ContainerLocation location, const std::vector<T> & data)
			:Container(location, data.size())
		{
#ifdef HAVE_CUDA
			if(location == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(this->get(), data.data(), this->size() * sizeof(T), cudaMemcpyHostToDevice, cudaStreamPerThread));
				cudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
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
		Container(ContainerLocation location, const T* dataBegin, const T* dataEnd)
			:Container(location, dataEnd - dataBegin)
		{
#ifdef HAVE_CUDA
			cudaSafeCall(cudaMemcpy(this->get(), dataBegin, this->size() * sizeof(T), cudaMemcpyDefault));
#else
			std::copy(data.begin(), data.end(), this->get());
#endif
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

		std::shared_ptr<Container<T> > getCopy(ContainerLocation newLocation) const
		{
#ifdef HAVE_CUDA
			auto ret = std::make_shared<Container<T> >(newLocation, this->size());
			if(m_location == LocationHost && newLocation == LocationHost)
			{
				std::copy(this->get(), this->get() + this->size(), ret->get());
			}
			else if(m_location == LocationHost && newLocation == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(ret->get(), this->get(), this->size() * sizeof(T), cudaMemcpyHostToDevice, cudaStreamPerThread));
				cudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
			}
			else if(m_location == LocationGpu && newLocation == LocationHost)
			{
				cudaSafeCall(cudaMemcpyAsync(ret->get(), this->get(), this->size() * sizeof(T), cudaMemcpyDeviceToHost, cudaStreamPerThread));
				cudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
			}
			else if(m_location == LocationGpu && newLocation == LocationGpu)
			{
				cudaSafeCall(cudaMemcpy(ret->get(), this->get(), this->size() * sizeof(T), cudaMemcpyDefault));
			}
			else {
				cudaSafeCall(cudaMemcpy(ret->get(), this->get(), this->size() * sizeof(T), cudaMemcpyDefault));
			}
			return ret;
#else
			return nullptr;
#endif
		}

		std::unique_ptr<Container<T> > getCopyUnique(ContainerLocation newLocation) const
		{
#ifdef HAVE_CUDA
			auto ret = std::unique_ptr<Container<T> >(new Container<T>(newLocation, this->size()));
			if(m_location == LocationHost && newLocation == LocationHost)
			{
				std::copy(this->get(), this->get() + this->size(), ret->get());
			}
			else if(m_location == LocationHost && newLocation == LocationGpu)
			{
				cudaSafeCall(cudaMemcpyAsync(ret->get(), this->get(), this->size() * sizeof(T), cudaMemcpyHostToDevice, cudaStreamPerThread));
				cudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
			}
			else if(m_location == LocationGpu && newLocation == LocationHost)
			{
				cudaSafeCall(cudaMemcpyAsync(ret->get(), this->get(), this->size() * sizeof(T), cudaMemcpyDeviceToHost, cudaStreamPerThread));
				cudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));				
			}
			else if(m_location == LocationGpu && newLocation == LocationGpu)
			{
				cudaSafeCall(cudaMemcpy(ret->get(), this->get(), this->size() * sizeof(T), cudaMemcpyDefault));
			}
			else 
			{
				cudaSafeCall(cudaMemcpy(ret->get(), this->get(), this->size() * sizeof(T), cudaMemcpyDefault));
			}
			return ret;
#else
			return nullptr;
#endif
		}

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
				cudaSafeCall(cudaMemcpyAsync(ret, this->get(), this->size() * sizeof(T), cudaMemcpyDeviceToHost, cudaStreamPerThread));
				cudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));				
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
	private:
		// The number of elements this container can store
		size_t m_numel;
		ContainerLocation m_location;

		T* m_buffer;
	};
}

#endif //!__CONTAINER_H__
