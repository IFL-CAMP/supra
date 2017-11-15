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

#include "WindowFunction.h"

using namespace std;

namespace supra
{
	WindowFunction::WindowFunction(WindowType type, ElementType windowParameter, size_t numEntriesPerFunction)
		: m_numEntriesPerFunction(numEntriesPerFunction)
		, m_scale(static_cast<float>(numEntriesPerFunction - 1)*0.5f)
		, m_type(type)
		, m_windowParameter(windowParameter)
		, m_gpuFunction(0, nullptr)
	{
		m_data.resize(m_numEntriesPerFunction);

		//compute the function values on the cpu
		std::vector<ElementType> function(m_numEntriesPerFunction);
		//compute the various functions
		ElementType maxValue = std::numeric_limits<ElementType>::min();
		for (size_t entryIdx = 0; entryIdx < m_numEntriesPerFunction; entryIdx++)
		{
			ElementType relativeEntryIdx = (static_cast<ElementType>(entryIdx) / (m_numEntriesPerFunction - 1))*2 - 1;
			ElementType value = windowFunction(m_type, relativeEntryIdx, m_windowParameter);
			m_data[entryIdx] = value;
			maxValue = std::max(maxValue, value);
		}
		// normalize window
		for (size_t entryIdx = 0; entryIdx < m_numEntriesPerFunction; entryIdx++)
		{
			m_data[entryIdx] /= maxValue;
		}

		//Create the storage for the window functions
		m_dataGpu = unique_ptr<Container<ElementType> >(
			new Container<ElementType>(LocationGpu, m_data));

		m_gpuFunction = WindowFunctionGpu(m_numEntriesPerFunction, m_dataGpu->get());
	}

	const WindowFunctionGpu* WindowFunction::getGpu() const
	{
		return &m_gpuFunction;
	}

	//Returns the weight of chosen window a the relative index
	// relativeIndex has to be normalized to [-1, 1] (inclusive)
	WindowFunction::ElementType WindowFunction::get(float relativeIndex) const
	{
		ElementType ret = 0;
		float relativeIndexClamped = min(max(relativeIndex, -1.0f), 1.0f);
		uint32_t absoluteIndex =
			static_cast<uint32_t>(std::round(m_scale*(relativeIndexClamped + 1.0f)));
		return m_data[absoluteIndex];
	}

	WindowFunction::ElementType WindowFunction::getDirect(uint32_t idx) const
	{
		ElementType ret = 0;
		if (idx < m_numEntriesPerFunction)
		{
			ret = m_data[idx];
		}
		return ret;
	}
}
