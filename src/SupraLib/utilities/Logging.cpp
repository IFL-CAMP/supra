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

#include "Logging.h"

#include <vector>
#include <cassert>
#include <cctype>
#include <ostream>
#include <sstream>
#include <memory>

using namespace std;

namespace supra
{
	ostream* logging::Base::sm_pOutStream = &cout;
	logging::SeverityMask logging::Base::sm_logLevel = logging::Severity::warning | logging::Severity::error | logging::Severity::always;
	std::mutex logging::Base::sm_streamMutex;
	std::ofstream logging::Base::sm_logFile;
	const std::string logging::Base::m_logFileName("supra.log");

	namespace logging
	{
		/// Specialized stringbuf that can be made the target of one of the standard streams
		/// such as cout. The output is then not directed to a file, but to the logging facilities.
		class RedirectionBuffer : public std::streambuf
		{
		public:
			/// Constructor
			explicit RedirectionBuffer() {};

			int overflow(int c);
			
		protected:
			/// Sends the contents of the buffer to the logging facilities
			void sendToLog();

		private:
			std::stringstream m_line;
		};

		/// This class uses two \see RedirectionBuffer to redirect cout and cerr to the log.
		/// It sets the redirection on construction and resets it on destruction.
		class CoutRedirector
		{
		public:
			/// Constructor. Sets the redirection of cout and cerr to the log.
			CoutRedirector()
			{
				m_originalCoutBuf = std::cout.rdbuf(); //save old buf
				m_originalCerrBuf = std::cerr.rdbuf(); //save old buf

				m_newCoutOstream = std::unique_ptr<std::ostream>(new ostream(m_originalCoutBuf));
				Base::setOutStream(m_newCoutOstream.get());

				m_externalBuffer = std::unique_ptr<RedirectionBuffer>(new RedirectionBuffer());
				m_externalCerrBuffer = std::unique_ptr<RedirectionBuffer>(new RedirectionBuffer());

				std::cout.rdbuf(m_externalBuffer.get()); //redirect std::cout
				std::cerr.rdbuf(m_externalCerrBuffer.get()); //redirect std::cerr			
			}

			/// Destructor. Resets the target buffers of cout and cerr.
			~CoutRedirector()
			{
				// undo redirections
				std::cerr.rdbuf(m_originalCerrBuf);
				std::cout.rdbuf(m_originalCoutBuf);

				Base::setOutStream(&std::cout);
			}

		private:
			std::streambuf * m_originalCerrBuf;
			std::streambuf * m_originalCoutBuf;
			std::unique_ptr<std::ostream> m_newCoutOstream;
			std::unique_ptr<RedirectionBuffer> m_externalBuffer;
			std::unique_ptr<RedirectionBuffer> m_externalCerrBuffer;
		};

		int RedirectionBuffer::overflow(int c)
		{
			char * s = pbase();
			size_t charsAdded = 0;
			while (s < pptr() && s < epptr())
			{
				char c_read = *s;
				if (c_read == '\n')
				{
					sendToLog();
				}
				else {
					m_line.put(c_read);
				}				
				s++;
				charsAdded++;
			}
			if (c == '\n')
			{
				sendToLog();
			}
			else {
				m_line.put(c);
			}
			setp(pbase(), epptr());
			return c;
		}

		void RedirectionBuffer::sendToLog()
		{
			Base::log(Severity::external, "External: ", m_line.str());
			m_line.clear();
			m_line.str(std::string());
		}

		/// The one instance of the CoutRedirector.
		/// Its scope is during the whole program runtime.
		CoutRedirector redir;
	}
}
