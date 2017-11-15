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

#ifndef __WINDOWFUNCTION_H__
#define __WINDOWFUNCTION_H__

#ifndef __CUDACC__
#include <algorithm>
#endif

#include <memory>
#include <Container.h>
#include <utilities/utility.h>
#include <utilities/cudaUtility.h>

namespace supra
{
#ifndef __CUDACC__
	using std::max;
	using std::min;
#else
	using ::max;
	using ::min;
#endif

	//forward declaration
	class WindowFunction;

	enum WindowType : uint32_t
	{
		WindowRectangular = 0,
		WindowHann = 1,
		WindowHamming = 2,
		WindowGauss = 3,
		WindowINVALID = 4
	};

	class WindowFunctionGpu
	{
	public:
		typedef float ElementType;

		WindowFunctionGpu(const WindowFunctionGpu& a)
			: m_numEntriesPerFunction(a.m_numEntriesPerFunction)
			, m_data(a.m_data)
			, m_scale(a.m_scale) {};

		//Returns the weight of chosen window a the relative index 
		// relativeIndex has to be normalized to [-1, 1] (inclusive)
		__device__ inline ElementType get(float relativeIndex) const
		{
			float relativeIndexClamped = min(max(relativeIndex, -1.0f), 1.0f);
			uint32_t absoluteIndex =
				static_cast<uint32_t>(roundf(m_scale*(relativeIndexClamped + 1.0f)));
			return m_data[absoluteIndex];
		}

		//Returns the weight of chosen window a the relative index
		// relativeIndex has to be normalized to [-1, 1] (inclusive)
		__device__ inline ElementType getShared(const ElementType * __restrict__ sharedData, float relativeIndex) const
		{
			float relativeIndexClamped = min(max(relativeIndex, -1.0f), 1.0f);
			uint32_t absoluteIndex =
				static_cast<uint32_t>(roundf(m_scale*(relativeIndexClamped + 1.0f)));
			return sharedData[absoluteIndex];
		}

		__device__ inline ElementType getDirect(uint32_t idx) const
		{
			ElementType ret = 0;
			if (idx < m_numEntriesPerFunction)
			{
				ret = m_data[idx];
			}
			return ret;
		}

		__device__ inline uint32_t numElements() const
		{
			return m_numEntriesPerFunction;
		}

	private:
		friend WindowFunction;
		__host__ WindowFunctionGpu(size_t numEntriesPerFunction, const ElementType* data)
			: m_numEntriesPerFunction(static_cast<uint32_t>(numEntriesPerFunction))
			, m_data(data)
			, m_scale(static_cast<float>(numEntriesPerFunction - 1)*0.5f) {};

		float m_scale;
		uint32_t m_numEntriesPerFunction;
		const ElementType* m_data;
	};

	class WindowFunction
	{
	public:
		typedef WindowFunctionGpu::ElementType ElementType;

		WindowFunction(WindowType type, ElementType windowParameter = 0.0, size_t numEntriesPerFunction = 128);

		const WindowFunctionGpu* getGpu() const;

		WindowType getType() const { return m_type; };
		ElementType getParameter() const { return m_windowParameter; };

		//Returns the weight of chosen window a the relative index
		// relativeIndex has to be normalized to [-1, 1] (inclusive)
		ElementType get(float relativeIndex) const;
		ElementType getDirect(uint32_t idx) const;

		// relativeIndex has to be normalized to [-1, 1] (inclusive)
		template <typename T>
		static __device__ __host__ inline T windowFunction(const WindowType& type, const T& relativeIndex, const T& windowParameter)
		{
			switch (type)
			{
			case WindowRectangular:
				return 1.0;
			case WindowHann:
				return (1 - windowParameter)*(0.5f - 0.5f*std::cos(2*static_cast<T>(M_PI)*((relativeIndex + 1) *0.5f))) + windowParameter;
			case WindowHamming:
				return (1 - windowParameter)*(0.54f - 0.46f*std::cos(2*static_cast<T>(M_PI)*((relativeIndex + 1) *0.5f))) + windowParameter;
			case WindowGauss:
				return static_cast<T>(
					1.0 / (windowParameter * sqrt(2.0 * M_PI)) * exp((-1.0 / 2.0) * (relativeIndex / windowParameter)*(relativeIndex / windowParameter)));
			default:
				return 0;
			}
		}
	private:
		size_t m_numEntriesPerFunction;
		std::vector<ElementType> m_data;
		std::unique_ptr<Container<ElementType> > m_dataGpu;
		ElementType m_scale;
		WindowType m_type;
		ElementType m_windowParameter;
		WindowFunctionGpu m_gpuFunction;
	};
}

#endif //!__WINDOWFUNCTION_H__
