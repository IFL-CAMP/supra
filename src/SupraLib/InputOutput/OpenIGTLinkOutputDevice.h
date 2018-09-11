// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2011-2016, all rights reserved,
//      Christoph Hennersperger 
//		EmaiL christoph.hennersperger@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
//	and
//		Rüdiger Göbl
//		Email r.goebl@tum.de
//
// ================================================================================================

#ifndef __OPENIGTLINKOUTPUTDEVICE_H__
#define __OPENIGTLINKOUTPUTDEVICE_H__

#ifdef HAVE_DEVICE_IGTL_OUTPUT

#include <tbb/flow_graph.h>

#include "AbstractOutput.h"
#include "USImage.h"
#include "SyncRecordObject.h"
#include "TrackerData.h"
#include "TrackerDataSet.h"

#include "igtlServerSocket.h"
#include "igtlClientSocket.h"
#include "igtlTrackingDataMessage.h"
#include "igtlImageMessage.h"

namespace supra
{
	class OpenIGTLinkOutputDevice : public AbstractOutput
	{
	public:
		OpenIGTLinkOutputDevice(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);
		~OpenIGTLinkOutputDevice();

		//Functions to be overwritten
	public:
		virtual void initializeOutput();
		virtual bool ready();
	protected:
		virtual void startOutput();
		//Needs to be thread safe
		virtual void stopOutput();
		//Needs to be thread safe
		virtual void configurationDone();

		virtual void writeData(std::shared_ptr<RecordObject> data);

	private:
		void sendMessage(std::shared_ptr<const RecordObject> data);
		void sendSyncRecordMessage(std::shared_ptr<const RecordObject> _syncMessage);
		template <typename T>
		void sendImageMessageTemplated(std::shared_ptr<const USImage> imageData);
		void sendImageMessage(std::shared_ptr<const RecordObject> imageData);
		void sendTrackingMessage(std::shared_ptr<const RecordObject> trackingData);
		void addTrackingData(igtl::TrackingDataMessage::Pointer msg, const TrackerData & trackerData, int targetSensor);

		void waitAsyncForConnection();

		igtl::ServerSocket::Pointer m_server;
		igtl::ClientSocket::Pointer m_clientConnection;

		bool m_isReady;
		std::atomic_bool m_isConnected;
		std::unique_ptr<std::thread> m_pConnectionThread;

		int m_port;
		std::string m_streamName;

	};
}

#endif //!HAVE_DEVICE_IGTL_OUTPUT

#endif //!__OPENIGTLINKOUTPUTDEVICE_H__
