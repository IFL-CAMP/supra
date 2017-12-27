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

#ifndef __TRACKERINTERFACEROS_H__
#define __TRACKERINTERFACEROS_H__

#ifdef HAVE_DEVICE_TRACKING_ROS

#include <AbstractInput.h>
#include <mutex>
#include <memory>

#include <geometry_msgs/TransformStamped.h>

namespace supra
{
	class RosWrapper;

	class TrackerInterfaceROS : public AbstractInput<RecordObject>
	{
	public:

		TrackerInterfaceROS(tbb::flow::graph & graph, const std::string & nodeID);
		~TrackerInterfaceROS() {};

		//Functions to be overwritten
	public:
		virtual void initializeDevice();
		virtual bool ready() { return true; };

		virtual std::vector<size_t> getImageOutputPorts() { return{}; };
		virtual std::vector<size_t> getTrackingOutputPorts() { return{ 0 }; };

		virtual void freeze();
		virtual void unfreeze();

	protected:
		virtual void startAcquisition();
		//Needs to be thread safe
		virtual void stopAcquisition() {};
		//Needs to be thread safe
		virtual void configurationEntryChanged(const std::string& configKey);
		//Needs to be thread safe
		virtual void configurationChanged();

	private:
		void receiveTransformMessage(const geometry_msgs::TransformStamped& transform);

		//double m_reconnectInterval;
		std::string m_rosMaster;
		std::string m_rosTopic;
		//uint32_t m_port;
		std::atomic<bool> m_frozen;

		std::mutex m_objectMutex;
		std::shared_ptr<RosWrapper> m_rosWrapper;
		bool m_connected;
	};
}

#endif //!HAVE_DEVICE_TRACKING_ROS

#endif //!__TRACKERINTERFACEROS_H__
