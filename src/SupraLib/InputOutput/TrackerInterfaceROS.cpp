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


#include "TrackerInterfaceROS.h"
#include "TrackerDataSet.h"

#include "utilities/RosWrapper.h"

#include <geometry_msgs/TransformStamped.h>

using namespace std;

namespace supra
{
	TrackerInterfaceROS::TrackerInterfaceROS(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractInput(graph, nodeID,1)
		, m_connected(false)
		, m_frozen(false)
	{
		m_valueRangeDictionary.set<string>("rosMaster", "", "Ros Master");
		m_valueRangeDictionary.set<string>("topic", "", "Ros Transform topic");
		m_valueRangeDictionary.set<int>("trackerID", 0, "Tracker ID");
		configurationChanged();
	}
	void TrackerInterfaceROS::initializeDevice()
	{
		logging::log_info("TrackerInterfaceROS:  Connecting to ros master at ", m_rosMaster);
		m_rosWrapper = make_shared<RosWrapper>(m_rosMaster);
	}

	void TrackerInterfaceROS::freeze()
	{
		m_frozen = true;
	}

	void TrackerInterfaceROS::unfreeze()
	{
		m_frozen = false;
	}

	void TrackerInterfaceROS::startAcquisition()
	{
		m_callFrequency.setName("TrROS");

		m_rosWrapper->subscribe(m_rosTopic, &TrackerInterfaceROS::receiveTransformMessage, this);
		m_rosWrapper->spin(&AbstractInput::getRunning, (AbstractInput<RecordObject>*)this);
	}

	void TrackerInterfaceROS::receiveTransformMessage(const geometry_msgs::TransformStamped & transform)
	{
		if (!m_frozen)
		{
			transform.header.stamp.toSec();

			auto pos = transform.transform.translation;
			auto quat = transform.transform.rotation;
			double timestamp = transform.header.stamp.toSec();

			std::vector<TrackerData> trackerData;
			trackerData.push_back(TrackerData(
			{ pos.x, pos.y, pos.z },
			{ quat.x, quat.y, quat.z, quat.w },
				100, m_trackerID,
				transform.child_frame_id,
				timestamp));

			auto pTrackingDataSet = make_shared<TrackerDataSet>(trackerData, timestamp, timestamp);
			addData<0>(pTrackingDataSet);
			m_callFrequency.measure();
		}
	}

	void TrackerInterfaceROS::configurationEntryChanged(const std::string & configKey)
	{
	}

	void TrackerInterfaceROS::configurationChanged()
	{
		//read conf values
		m_rosMaster = m_configurationDictionary.get<string>("rosMaster");
		m_rosTopic = m_configurationDictionary.get<string>("topic");
		m_trackerID = m_configurationDictionary.get<int>("trackerID");
	}
}
