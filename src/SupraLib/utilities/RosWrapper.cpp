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

#include "RosWrapper.h"

#include <memory>

#ifdef ROS_ROSSERIAL
#undef ERROR
#include <ros.h>
#else
#include <cstdlib>
#include <ros/ros.h>
#endif

using namespace std;

namespace supra
{
	RosWrapper::RosWrapper(const std::string & rosMasterHostname)
		: m_rosMasterHostname(rosMasterHostname)
	{
#ifdef ROS_ROSSERIAL
		char* rosMasterChar = new char[m_rosMasterHostname.length() + 1];
		strcpy(rosMasterChar, m_rosMasterHostname.c_str());

		m_rosNode = shared_ptr<ros::NodeHandle>(new ros::NodeHandle());
		m_rosNode->initNode(rosMasterChar);

		delete[] rosMasterChar;
#else
		string masterUri = "http://" + m_rosMasterHostname + ":11311/";
		setenv("ROS_MASTER_URI", masterUri.c_str(), true);

		int argc = 0;
		ros::init(argc, (char**)nullptr, "supra", ros::init_options::AnonymousName);
		m_rosNode = shared_ptr<ros::NodeHandle>(new ros::NodeHandle());
		m_callbackQueue = unique_ptr<ros::CallbackQueue>(new ros::CallbackQueue());
		m_rosNode->setCallbackQueue(m_callbackQueue.get());
#endif
	}
}
