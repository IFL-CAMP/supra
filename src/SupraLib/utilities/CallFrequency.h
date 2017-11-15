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

#ifndef __CALLFREQUENCY_H__
#define __CALLFREQUENCY_H__

#include <string>
#include <iostream>
#include <mutex>

#ifndef LOG_FREQUENCIES
#define LOG_FREQUENCIES true
#endif //!LOG_FREQUENCIES
#ifndef LOG_PROFILING
#define LOG_PROFILING false
#endif //!LOG_PROFILING

namespace supra
{
	/*! \brief Helper class for performance measurements.
	*          Logs the call frequency every second with a log-level "Log".
	*
	*  This class can be used to measure how frequent a particular line is executed.
	*  Additionally prints the total number of calls on destruction.
	*
	*  Example usage:
	*  @code
	*  void functionToMeasure(...)
	*  {
	*      static CallFrequency m('Identification String');
	*      m.measure();
	*      <expensive operation>
	*	   m.measureEnd();
	*  }
	*  @endcode
	*/
	class CallFrequency {
	public:
		/// Constructor of CallFrequency that takes the name
		CallFrequency(std::string name);
		/// Default constructor of CallFrequency
		CallFrequency();
		~CallFrequency();
		/// Method that should be called directly before a repeating operation
		/// that is to be measured. Samples the frequency of the calls and is part
		/// of the runtime measurement.
		void measure();
		/// Method that should be called directly after a repeating operation whose
		/// runtime should be measured. If this method is not called, only the 
		/// frequency of calls can be recorded.
		void measureEnd();
		/// Logs the current frequency and runtime measurements (if available)
		void print();
		/// Returns a string with the current frequency and runtime measurements (if available)
		std::string getTimingInfo();
		/// Sets the display name of this instance. 
		/// The name is used when logging the timings to distinguish them.
		void setName(std::string name);

	private:
		static void printFrequency(std::string name, double frequency);
		static void printFrequencyAndRuntime(std::string name, double frequency, double runtime);
		static void printEnd(std::string name, unsigned int callNum);

		const double m_fFilter = 0.025;
		std::string m_name;
		double m_flastTime;
		double m_ffiltTimeDelta;
		double m_filtRuntime;
		double m_lastPrint;
		unsigned int m_callNum;
		bool m_initialized;
		bool m_runTimeInitialized;
	};
}

#endif // !__CALLFREQUENCY_H__
