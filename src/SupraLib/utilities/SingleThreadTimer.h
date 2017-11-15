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

#ifndef __SINGLETHREADTIMER_H__
#define __SINGLETHREADTIMER_H__

#include <chrono>

namespace supra
{
	/// A simple timer that provides approximative intervals by yielding the thread
	class SingleThreadTimer {
	public:
		/// Returns the current timer frequency 
		double getFrequency();
		/// Configures the timer frequency
		void setFrequency(double frequency);
		/// Yields the thread until the beginning of the next timeslot as defined by
		/// the configured frequency. As it uses operating system timers, it is only
		/// approximate and there are no guarantees.
		void sleepUntilNextSlot();
	private:
		typedef std::chrono::high_resolution_clock clock;
		typedef std::chrono::duration<long, std::micro> duration;

		double m_frequency;
		duration m_slotDuration;
		clock::time_point m_beginLastSlot;
		bool m_initialized;
		bool m_firstRunDone;
	};
}

#endif // !__SINGLETHREADTIMER_H__