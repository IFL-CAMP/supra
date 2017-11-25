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

#ifndef __LOGGING_H__
#define __LOGGING_H__

#include <mutex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <type_traits>
#include <atomic>
#include <array>
#include <ctime>

#ifdef WIN32 
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#endif

namespace supra
{
	/*! \brief The main logging facility.
	*/
	namespace logging
	{
		/// Tags the severity of a message.
		typedef int SeverityMask;
		/// Tags the severity of a message.
		enum Severity : SeverityMask {
			log = 1,  ///Messages that might be of interest when understanding operation surrounding an error. Such events can happen repeatedly. E.g. call frequencies.
			info = 2,  ///A message tagged with info descibes non-repeating events that are handled correctly. E.g. start/stop of interfaces.
			warning = 4,  ///Warn the user of circumstances that should not arise normally. Use this level if there is a good chance of continued operation.
			error = 8,  ///The message describes an unexpected failure. Normal operation probably cannot be kept up.
			param = 16,  ///The message describes an parameter change.
			profiling = 32, ///The message serves the profiling of the application
			external = 64,  ///Message coming from an external lib
			always = 128   ///Message has to be printed always
		};

#ifdef WIN32
		/// Stream modifier for blue text
		inline std::ostream& blue(std::ostream &s)
		{
			HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(h,
				FOREGROUND_BLUE | FOREGROUND_INTENSITY);
			return s;
		}
		/// Stream modifier for red text
		inline std::ostream& red(std::ostream &s)
		{
			HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(h,
				FOREGROUND_RED | FOREGROUND_INTENSITY);
			return s;
		}
		/// Stream modifier for green text
		inline std::ostream& green(std::ostream &s)
		{
			HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(h,
				FOREGROUND_GREEN | FOREGROUND_INTENSITY);
			return s;
		}
		/// Stream modifier for yellow text
		inline std::ostream& yellow(std::ostream &s)
		{
			HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(h,
				FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);
			return s;
		}
		/// Stream modifier to reset text color
		inline std::ostream& reset(std::ostream &s)
		{
			HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(h,
				FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
			return s;
		}
#else
		/// Stream modifier for red text
		const std::string red("\033[0;31m");
		/// Stream modifier for green text
		const std::string green("\033[1;32m");
		/// Stream modifier for yellow text
		const std::string yellow("\033[1;33m");
		/// Stream modifier for cyan text
		const std::string cyan("\033[0;36m");
		/// Stream modifier for magenta text
		const std::string magenta("\033[0;35m");
		/// Stream modifier to reset text color
		const std::string reset("\033[0m");
#endif
		/// Endline character to use for logging
		const std::string endl("\n");

		/// Logging base functionality
		class Base
		{
		public:
			/// Internal: Implementation of logging output
			template <typename... outObjectTypes>
			static void log(outObjectTypes... o)
			{
				//acquire mutex to make it legible
				std::lock_guard<std::mutex> lock(sm_streamMutex);
				logRec(o...);
				logRecFile(o...);
			}

			/// Internal: Implementation of logging output with severity
			template <typename... outObjectTypes>
			static void log(Severity severity, outObjectTypes... o)
			{
				//acquire mutex to make it legible
				std::lock_guard<std::mutex> lock(sm_streamMutex);
				if ((severity & sm_logLevel) > 0)
				{
					logRec(o..., endl);
				}
				logRecFile(o..., endl);
			}

			/// Sets the output stream for the console log output
			static void setOutStream(std::ostream* newOut)
			{
				std::lock_guard<std::mutex> lock(sm_streamMutex);
				sm_pOutStream->flush();
				sm_pOutStream = newOut;
			}

			/// Sets which log-levels are shown on the console log output.
			/// Accepts a mask built from values of \see SeverityMask
			/// default: Severity::warning | Severity::error | Severity::always
			static void setLogLevel(SeverityMask severity)
			{
				sm_logLevel = severity | logging::Severity::always;
			}
		private:
			template <typename firstObjectType>
			static void logRec(firstObjectType first)
			{
				if (std::is_arithmetic<firstObjectType>::value)
				{
					(*sm_pOutStream) << std::setbase(10) << first;
				}
				else {
					(*sm_pOutStream) << first;
				}
			}

			template <typename firstObjectType, typename... outObjectTypes>
			static void logRec(firstObjectType first, outObjectTypes... o)
			{
				logRec(first);
				logRec(o...);
			}

			template <typename firstObjectType>
			static void logRecFile(firstObjectType first)
			{
				if (!sm_logFile.is_open())
				{
					initLogfile();
				}

				if (std::is_arithmetic<firstObjectType>::value)
				{
					sm_logFile << std::setbase(10);
					if (std::is_floating_point<firstObjectType>::value)
					{
						sm_logFile << std::setprecision(std::numeric_limits<firstObjectType>::digits10 + 1);
					}
					sm_logFile << first;
					
				}
				else {
					sm_logFile << first;
				}
			}

			template <typename firstObjectType, typename... outObjectTypes>
			static void logRecFile(firstObjectType first, outObjectTypes... o)
			{
				logRecFile(first);
				logRecFile(o...);
			}

			static void initLogfile()
			{
				sm_logFile.open(m_logFileName, std::ios_base::out | std::ios_base::app);

				std::array<char, 64> buffer;
				buffer.fill(0);
				std::time_t rawtime;
				std::time(&rawtime);
				const auto timeinfo = std::localtime(&rawtime);
				std::strftime(buffer.data(), sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);

				sm_logFile << std::endl << std::endl <<
					"------------------------------------------" << std::endl <<
					"SUPRA (" << buffer.data() << ")" << std::endl;
			}

			static std::mutex sm_streamMutex;
			static std::ostream* sm_pOutStream;
			static SeverityMask sm_logLevel;
			static std::ofstream sm_logFile;
			static const std::string m_logFileName;
		};

		/// Log entry that has always to be shown.
		template <typename... outObjectTypes>
		void log_always(outObjectTypes... o)
		{
			Base::log(Severity::always, o...);
		}

		/// Log entry with the lowest severity, is only logged if cond is true
		/// Messages that might be of interest when understanding operation surrounding an error. Such events can happen repeatedly. E.g. call frequencies.
		template <typename... outObjectTypes>
		void log_log_if(bool cond, outObjectTypes... o)
		{
			if (cond)
			{
				Base::log(Severity::log, o...);
			}
		}

		/// Log entry that informs the user of events of normal operation, is only logged if cond is true
		/// A message tagged with info descibes non-repeating events that are handled correctly. E.g. start/stop of interfaces.
		template <typename... outObjectTypes>
		void log_info_if(bool cond, outObjectTypes... o)
		{
			if (cond)
			{
				Base::log(Severity::info, o...);
			}
		}

		/// Log entry representing a warning, is only logged if cond is true
		/// Warn the user of circumstances that should not arise normally. Use this level if there is a good chance of continued operation.
		template <typename... outObjectTypes>
		void log_warn_if(bool cond, outObjectTypes... o)
		{
			if (cond)
			{
				Base::log(Severity::warning, yellow, o..., reset);
			}
		}

		/// Log entry representing an error, is only logged if cond is true
		/// The message describes an unexpected failure. Normal operation probably cannot be kept up.
		template <typename... outObjectTypes>
		void log_error_if(bool cond, outObjectTypes... o)
		{
			if (cond)
			{
				Base::log(Severity::error, red, o..., reset);
			}
		}
		/// Log entry containing profiling details
		/// Should only be used internally by the CallFrequency
		template <typename... outObjectTypes>
		void log_profiling_if(bool cond, outObjectTypes... o)
		{
			if (cond)
			{
				Base::log(Severity::profiling, o...);
			}
		}

		/// Log entry with the lowest severity
		/// Messages that might be of interest when understanding operation surrounding an error. Such events can happen repeatedly. E.g. call frequencies.
		template <typename... outObjectTypes>
		void log_log(outObjectTypes... o)
		{
			log_log_if(true, o...);
		}
		/// Log entry that informs the user of events of normal operation
		/// A message tagged with info descibes non-repeating events that are handled correctly. E.g. start/stop of interfaces.
		template <typename... outObjectTypes>
		void log_info(outObjectTypes... o)
		{
			log_info_if(true, o...);
		}
		/// Log entry representing a warning
		/// Warn the user of circumstances that should not arise normally. Use this level if there is a good chance of continued operation.
		template <typename... outObjectTypes>
		void log_warn(outObjectTypes... o)
		{
			log_warn_if(true, o...);
		}
		/// Log entry representing an error
		/// The message describes an unexpected failure. Normal operation probably cannot be kept up.
		template <typename... outObjectTypes>
		void log_error(outObjectTypes... o)
		{
			log_error_if(true, o...);
		}
		/// Log entry informing about the change of a node parameter
		/// Should only be used internally by the parameter system
		template <typename... outObjectTypes>
		void log_parameter(outObjectTypes... o)
		{
			Base::log(Severity::param, o...);
		}
		/// Log entry containing profiling details
		/// Should only be used internally by the CallFrequency
		template <typename... outObjectTypes>
		void log_profiling(outObjectTypes... o)
		{
			log_profiling_if(true, o...);
		}
	}
}

#endif // !__LOGGING_H__
