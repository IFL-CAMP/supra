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

#include "SingleThreadTimer.h"

#include <thread>
#include <cmath>

using namespace std;
using namespace std::chrono;

namespace supra
{
	double SingleThreadTimer::getFrequency()
	{
		return m_frequency;
	}

	void SingleThreadTimer::setFrequency(double frequency)
	{
		m_frequency = frequency;
		m_beginLastSlot = clock::now();
		m_slotDuration = duration_cast<duration>(microseconds((long long)round(1e6 / frequency)));
		m_initialized = true;
		m_firstRunDone = false;
	}

	void SingleThreadTimer::sleepUntilNextSlot()
	{
		if (m_initialized)
		{
			if (m_firstRunDone)
			{
				//compute how long this thread needs to sleep
				clock::time_point beginNextSlot = m_beginLastSlot + m_slotDuration;

				//sleep
				this_thread::sleep_until(beginNextSlot);
			}
			else
			{
				m_firstRunDone = true;
			}
			m_beginLastSlot = clock::now();
		}
	}
}