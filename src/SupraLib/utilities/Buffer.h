// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __BUFFER_H__
#define __BUFFER_H__

#include "vec.h"
#include <type_traits>

namespace supra
{

#ifdef HAVE_CUDA
	template <typename ElementPtrType, typename IndexType>
	class Buffer2
	{
	public:
		typedef typename std::remove_pointer<ElementPtrType>::type ElementType;
		
		static_assert(std::is_pointer<ElementPtrType>::value, "Buffer2 requires a pointer type");

	public:
		__device__ __host__ Buffer2(ElementPtrType buffer, vec2T<IndexType> size)
			: m_buffer{ buffer }
			, m_size{ size } { };

		__device__ __host__ ElementType& operator[](vec2T<IndexType> index)
		{
			return accessBuffer(index);
		}

	protected:
		__device__ __host__ ElementType& accessBuffer(vec2T<IndexType> index)
		{
			return m_buffer[index.x + index.y * m_size.x];
		}

		ElementPtrType m_buffer;
		vec2T<IndexType> m_size;
	};

	template <typename ElementPtrType, typename IndexType>
	class Buffer3
	{
	public:
		typedef typename std::remove_pointer<ElementPtrType>::type ElementType;

		static_assert(std::is_pointer<ElementPtrType>::value, "Buffer3 requires a pointer type");

	public:
		__device__ __host__ Buffer3(ElementPtrType buffer, vec3T<IndexType> size)
			: m_buffer{ buffer }
			, m_size{ size } { };

		__device__ __host__ ElementType& operator[](vec3T<IndexType> index)
		{
			return accessBuffer(index);
		}

	protected:
		__device__ __host__ ElementType& accessBuffer(vec3T<IndexType> index)
		{
			return m_buffer[index.x + index.y * m_size.x + index.z * m_size.x*m_size.y];
		}

		ElementPtrType m_buffer;
		vec3T<IndexType> m_size;
	};

	template <typename ElementPtrType, typename IndexType>
	class CachedBuffer2 : Buffer2<ElementPtrType, IndexType>
	{
	public:
		typedef typename std::remove_const<std::remove_pointer<ElementPtrType>::type>::type ModifiableElementType;
		typedef typename std::add_pointer<ModifiableElementType>::type ModifiableElementPtrType;

	public:
		__device__ CachedBuffer2(
			ElementPtrType buffer, vec2T<IndexType> size,
			ModifiableElementPtrType cacheBuffer, vec2T<IndexType> cacheSize,
			vec2T<IndexType> cacheOffset)
			: Buffer2(buffer, size)
			, m_cachedBuffer{ cacheBuffer }
			, m_cacheSize{ cacheSize }
			, m_cacheOffset{ cacheOffset }
			, m_cacheEnd{ cacheOffset + cacheSize }
		{
			loadIntoCache();
		};

		__device__ ElementType& operator[](vec2T<IndexType> index)
		{
			// If the element is cached give the caller access to the cache
			if (index.x >= m_cacheOffset.x && index.x < m_cacheEnd.x &&
				index.y >= m_cacheOffset.y && index.y < m_cacheEnd.y)
			{
				return accessCache(index);
			}
			else // If the element is not cached, refer to the uncached buffer
			{
				return accessBuffer(index);
			}
		}

		__device__ void writeCacheToBuffer()
		{
			// Copy the data in a grid stride loop
			// ( because the cache usually is bigger than the block)
			vec2T<IndexType> index{ m_cacheOffset };
			for (; index.y < m_cacheEnd.y; index.y += blockDim.y)
			{
				for (; index.x < m_cacheEnd.x; index.x += blockDim.x)
				{
					accessBuffer(index) = accessCache(index);
				}
			}
			__syncthreads();
		};

	protected:

		__device__ ModifiableElementType& accessCache(vec2T<IndexType> index)
		{
			return m_cachedBuffer[
				(index.x - m_cacheOffset.x) +
					(index.y - m_cacheOffset.y) * m_cacheSize.x];
		}

		__device__ void loadIntoCache()
		{
			// Copy the data in a grid stride loop
			// ( because the cache usually is bigger than the block)
			vec2T<IndexType> index{ m_cacheOffset + vec2T<IndexType>{threadIdx.x, threadIdx.y} };
			
			for (; index.y < m_cacheEnd.y; index.y += blockDim.y)
			{
				for (; index.x < m_cacheEnd.x; index.x += blockDim.x)
				{
					accessCache(index) = accessBuffer(index);
				}
			}
			__syncthreads();
		};

		ModifiableElementPtrType m_cachedBuffer;
		vec2T<IndexType> m_cacheSize;
		vec2T<IndexType> m_cacheOffset;
		vec2T<IndexType> m_cacheEnd;
	};
	
	template <typename ElementPtrType, typename IndexType>
	class CachedBuffer3 : Buffer3<ElementPtrType, IndexType>
	{
	public:
		typedef typename std::remove_const<std::remove_pointer<ElementPtrType>::type>::type ModifiableElementType;
		typedef typename std::add_pointer<ModifiableElementType>::type ModifiableElementPtrType;

	public:
		__device__ CachedBuffer3(
			ElementPtrType buffer, vec3T<IndexType> size, 
			ModifiableElementPtrType cacheBuffer, vec3T<IndexType> cacheSize,
			vec3T<IndexType> cacheOffset)
			: Buffer3(buffer, size)
			, m_cachedBuffer { cacheBuffer }
			, m_cacheSize{ cacheSize }
			, m_cacheOffset{ cacheOffset }
			, m_cacheEnd{ cacheOffset + cacheSize } 
		{
			loadIntoCache();
		};

		__device__ ElementType& operator[](vec3T<IndexType> index)
		{
			// If the element is cached give the caller access to the cache
			if (index.x >= m_cacheOffset.x && index.x < m_cacheEnd.x &&
				index.y >= m_cacheOffset.y && index.y < m_cacheEnd.y &&
				index.z >= m_cacheOffset.z && index.z < m_cacheEnd.z)
			{
				return accessCache(index);
			}
			else // If the element is not cached, refer to the uncached buffer
			{
				return accessBuffer(index);
			}
		}

		__device__ void writeCacheToBuffer()
		{
			// Copy the data in a grid stride loop
			// ( because the cache usually is bigger than the block)
			vec3T<IndexType> index{ m_cacheOffset };
			for (; index.z < m_cacheEnd.z; index.z += blockDim.z)
			{
				for (; index.y < m_cacheEnd.y; index.y += blockDim.y)
				{
					for (; index.x < m_cacheEnd.x; index.x += blockDim.x)
					{
						accessBuffer(index) = accessCache(index);
					}
				}
			}
			__syncthreads();
		};

	protected:

		__device__ ModifiableElementType& accessCache(vec3T<IndexType> index)
		{
			return m_cachedBuffer[
				(index.x - m_cacheOffset.x) + 
				(index.y - m_cacheOffset.y) * m_cacheSize.x + 
				(index.z - m_cacheOffset.z) * m_cacheSize.x * m_cacheSize.y];
		}

		__device__ void loadIntoCache()
		{
			// Copy the data in a grid stride loop
			// ( because the cache usually is bigger than the block)
			vec3T<IndexType> index{ m_cacheOffset + vec3T<IndexType>{threadIdx.x, threadIdx.y, threadIdx.z} };
			for (; index.z < m_cacheEnd.z; index.z += blockDim.z)
			{
				for (; index.y < m_cacheEnd.y; index.y += blockDim.y)
				{
					for (; index.x < m_cacheEnd.x; index.x += blockDim.x)
					{
						accessCache(index) = accessBuffer(index);
					}
				}
			}
			__syncthreads();
		};

		ModifiableElementPtrType m_cachedBuffer;
		vec3T<IndexType> m_cacheSize;
		vec3T<IndexType> m_cacheOffset;
		vec3T<IndexType> m_cacheEnd;
	};

#endif //HAVE_CUDA
}

#endif // !__BUFFER_H__
