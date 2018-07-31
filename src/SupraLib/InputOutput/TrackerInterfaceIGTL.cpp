// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016-2017, all rights reserved,
//      Rüdiger Göbl
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
//
// ================================================================================================


#include "TrackerInterfaceIGTL.h"

#include <igtlTrackingDataMessage.h>
//#include <utilities/utility.h>
#include <thread>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

namespace supra
{
	TrackerInterfaceIGTL::TrackerInterfaceIGTL(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractInput<RecordObject>(graph, nodeID)
		, m_connected(false)
		, m_frozen(false)
	{
		m_valueRangeDictionary.set<double>("reconnectInterval", 0.01, 3600, 0.1, "Reconnect Interval [s]");
		m_valueRangeDictionary.set<string>("hostname", "", "Server hostname");
		m_valueRangeDictionary.set<uint32_t>("port", 1, 65535, 18944, "Server port");
		configurationChanged();
	}
	void TrackerInterfaceIGTL::initializeDevice()
	{
		//try to connect already here, so we are directly good to go!
		lock_guard<mutex> lock(m_objectMutex);
		m_socket = igtl::ClientSocket::New();
		connectToSever();
	}

	void TrackerInterfaceIGTL::freeze()
	{
		m_frozen = true;
	}

	void TrackerInterfaceIGTL::unfreeze()
	{
		m_frozen = false;
	}

	void TrackerInterfaceIGTL::startAcquisition()
	{
		m_callFrequency.setName("TrIGTL");

		while (getRunning())
		{
			if (!m_connected)
			{
				lock_guard<mutex> lock(m_objectMutex);

				connectToSever();
			}

			//------------------------------------------------------------
			// Wait for a reply
			if (m_connected)
			{
				igtl::MessageHeader::Pointer headerMsg;
				headerMsg = igtl::MessageHeader::New();
				headerMsg->InitPack();
				int rs = m_socket->Receive(headerMsg->GetPackPointer(), headerMsg->GetPackSize());
				{
					lock_guard<mutex> lock(m_objectMutex);
					if (rs == 0)
					{
						logging::log_warn("TrackerInterfaceIGTL: Connection closed.");
						closeSocket();
						continue;
					}
					if (rs != headerMsg->GetPackSize())
					{
						logging::log_warn("TrackerInterfaceIGTL: Message size information and actual data size don't match.");
						closeSocket();
						continue;
					}

					if (!m_frozen)
					{
						headerMsg->Unpack();
						if (strcmp(headerMsg->GetDeviceType(), "TDATA") == 0)
						{
							receiveTrackingData(headerMsg);
						}
						else
						{
							m_socket->Skip(headerMsg->GetBodySizeToRead(), 0);
						}
					}
				}
			}
			else {
				logging::log_warn("TrackerInterfaceIGTL: Could not reconnect to the server '", m_hostname, ":", m_port, "'. Retrying in ", m_reconnectInterval, "s.");
				duration<long, std::milli> sleepDuration = milliseconds((long long)round(m_reconnectInterval*1e3));
				this_thread::sleep_for(sleepDuration);
			}
		}

		{
			lock_guard<mutex> lock(m_objectMutex);
			closeSocket();
		}

	}

	void TrackerInterfaceIGTL::connectToSever()
	{
		if (!m_connected)
		{
			int r = m_socket->ConnectToServer(m_hostname.c_str(), m_port);

			if (r != 0)
			{
				m_connected = false;
				logging::log_warn("TrackerInterfaceIGTL: Could not reconnect to the server '", m_hostname, ":", m_port, "'");
			}
			else {
				m_connected = true;
				logging::log_info("TrackerInterfaceIGTL: Connected to the server '", m_hostname, ":", m_port, "'");
			}
		}
	}

	void TrackerInterfaceIGTL::closeSocket()
	{
		m_connected = false;
		logging::log_warn("TrackerInterfaceIGTL: Closing socket to the server '", m_hostname, ":", m_port, "'");
		m_socket->CloseSocket();
	}

	bool TrackerInterfaceIGTL::receiveTrackingData(igtl::MessageHeader::Pointer& header)
	{
		//------------------------------------------------------------
		// Allocate TrackingData Message Class

		igtl::TrackingDataMessage::Pointer trackingData;
		trackingData = igtl::TrackingDataMessage::New();
		trackingData->SetMessageHeader(header);
		trackingData->AllocatePack();

		// Receive body from the socket
		m_socket->Receive(trackingData->GetPackBodyPointer(), trackingData->GetPackBodySize());

		// Deserialize the transform data
		// If you want to skip CRC check, call Unpack() without argument.
		int c = trackingData->Unpack(1);

		bool crcFine = (c & igtl::MessageHeader::UNPACK_BODY) > 0;

		if (crcFine) // if CRC check is OK
		{
			std::vector<TrackerData> trackerData;

			//compute float timestamp format from IGTL representation
			uint32_t timestampSeconds;
			uint32_t timestampFrac;
			trackingData->GetTimeStamp(&timestampSeconds, &timestampFrac);
			double timestamp = (double)timestampSeconds + ((double)timestampFrac) / 1e9;

			int nElements = trackingData->GetNumberOfTrackingDataElements();
			for (int i = 0; i < nElements; i++)
			{
				igtl::TrackingDataElement::Pointer trackingElement;
				trackingData->GetTrackingDataElement(i, trackingElement);

				igtl::Matrix4x4 igtlMatrix;
				trackingElement->GetMatrix(igtlMatrix);

				TrackerData::Matrix matrix;
				matrix[0 + 0] = igtlMatrix[0][0];
				matrix[0 + 1] = igtlMatrix[0][1];
				matrix[0 + 2] = igtlMatrix[0][2];
				matrix[0 + 3] = igtlMatrix[0][3];

				matrix[4 + 0] = igtlMatrix[1][0];
				matrix[4 + 1] = igtlMatrix[1][1];
				matrix[4 + 2] = igtlMatrix[1][2];
				matrix[4 + 3] = igtlMatrix[1][3];

				matrix[8 + 0] = igtlMatrix[2][0];
				matrix[8 + 1] = igtlMatrix[2][1];
				matrix[8 + 2] = igtlMatrix[2][2];
				matrix[8 + 3] = igtlMatrix[2][3];

				matrix[12 + 0] = igtlMatrix[3][0];
				matrix[12 + 1] = igtlMatrix[3][1];
				matrix[12 + 2] = igtlMatrix[3][2];
				matrix[12 + 3] = igtlMatrix[3][3];


				trackerData.push_back(TrackerData(matrix, 100, 666, trackingElement->GetName(), timestamp));
			}

			auto pTrackingDataSet = make_shared<TrackerDataSet>(trackerData, timestamp, timestamp);
			addData<0>(pTrackingDataSet);
			m_callFrequency.measure();
		}
		else {
			logging::log_warn("TrackerInterfaceIGTL: IGTL message CRC error, skipping message");
		}

		return crcFine;
	}

	void TrackerInterfaceIGTL::configurationEntryChanged(const std::string & configKey)
	{
		std::lock_guard<std::mutex> lock(m_objectMutex);

		if (configKey == "reconnectInterval")
		{
			m_reconnectInterval = m_configurationDictionary.get<double>("reconnectInterval");
		}

		bool reconnectNeccessary = false;
		if (configKey == "hostname")
		{
			m_hostname = m_configurationDictionary.get<string>("hostname");
			reconnectNeccessary = true;
		}
		if (configKey == "port")
		{
			m_port = m_configurationDictionary.get<int>("port");
			reconnectNeccessary = true;
		}

		if (reconnectNeccessary)
		{
			closeSocket();
		}
	}

	void TrackerInterfaceIGTL::configurationChanged()
	{
		//read conf values
		m_reconnectInterval = m_configurationDictionary.get<double>("reconnectInterval");

		m_hostname = m_configurationDictionary.get<string>("hostname");
		m_port = m_configurationDictionary.get<uint32_t>("port");
	}
}
