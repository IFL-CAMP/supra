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

#ifndef __ROSWRAPPER_H__
#define __ROSWRAPPER_H__

#ifdef ROS_PRESENT

#include <utilities/Logging.h>
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <type_traits>
#include <cassert>
#include <vector>

#ifdef ROS_ROSSERIAL
#undef ERROR
#include <ros.h>
namespace ros
{
	class Subscriber_;
}
typedef ros::Subscriber_ SubscriberType;
#else
#include <ros/ros.h>
#include <ros/callback_queue.h>
namespace ros
{
	class Subscriber;
}
typedef ros::Subscriber SubscriberType;
#endif

namespace supra
{
#ifdef ROS_ROSSERIAL
#define ROSSERIAL_MSG_STRING(_x_) (stringToNewCstr(_x_))
#else
#define ROSSERIAL_MSG_STRING(_x_) (_x_)
#endif

#ifdef ROS_ROSSERIAL
#define ROSSERIAL_MSG_SETUP_ARRAY(_arr_, _length_) \
		do { \
			(##_arr_##_length) = static_cast<int>(_length_); \
			_arr_ = new std::remove_reference<decltype(*_arr_)>::type[_length_]; \
		} while(0);

#else
#define ROSSERIAL_MSG_SETUP_ARRAY(_arr_, _length_) \
		do{ _arr_.resize(_length_); } while(0);
#endif

	/// A wrapper around the functionality of publishing and advertising services 
	/// from a node in ROS on linux and rosserial on windows
	class RosWrapper {
	public:
		/// Constructor. The hostname of the ros master hostname needs to passed.
		RosWrapper(const std::string & rosMasterHostname);

		/// Returns the node handle created internally.
		std::shared_ptr<ros::NodeHandle> getNodeHandle()
		{
			return m_rosNode;
		}

		/// Subscribe to a topic. Follows ROS conventions
		template <class M, class T>
		void subscribe(
			const std::string & topic,
			void(T::*fp)(const M &), T* obj)
		{
#ifdef ROS_ROSSERIAL
			m_subscriber = std::unique_ptr<ros::Subscriber < M, T> >(
				new ros::Subscriber <M, T>(topic.c_str(), fp, obj)
				);
			m_rosNode->subscribe(*m_subscriber);
#else
			m_subscriber = std::unique_ptr<ros::Subscriber>(new ros::Subscriber());
			*m_subscriber = m_rosNode->subscribe(topic, 1, fp, obj);
#endif
		}

		/// Advertise a service
		template <class M>
		size_t advertise(const std::string & topic)
		{
#ifdef ROS_ROSSERIAL
			logging::log_error("Rosserial publishers not implemented yet");
#else
			auto pub = std::shared_ptr<ros::Publisher>(new ros::Publisher());
			*pub = m_rosNode->advertise<M>(topic, 1);
			m_publishers.push_back(pub);

			return m_publishers.size() - 1;
#endif
		}

		/// Publish to a topic
		template <class M>
		void publish(const M& msg, size_t publisherNo)
		{
			assert(publisherNo < m_publishers.size());
#ifdef ROS_ROSSERIAL
			logging::log_error("Rosserial publishers not implemented yet");
#else
			m_publishers[publisherNo]->publish(msg);
#endif
		}

		/// Spins the node until the callback function returns false
		template <class T>
		void spin(bool(T::*fp)(), T* obj)
		{
#ifdef ROS_ROSSERIAL
			while ((*obj.*fp)())
			{
				m_rosNode->spinOnce();
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
#else
			while ((*obj.*fp)() && m_rosNode->ok())
			{
				m_callbackQueue->callAvailable(ros::WallDuration(0.1));
			}
#endif
		}

		/// Spins the node
		void spinEndless()
		{
#ifdef ROS_ROSSERIAL
			while (true)
			{
				m_rosNode->spinOnce();
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
#else
			while (m_rosNode->ok())
			{
				m_callbackQueue->callAvailable(ros::WallDuration(0.1));
			}
#endif
		}

	private:
		std::string m_rosMasterHostname;
		std::shared_ptr<ros::NodeHandle> m_rosNode;
		std::unique_ptr<SubscriberType> m_subscriber;
		std::vector<std::shared_ptr<ros::Publisher> > m_publishers;
#ifdef ROS_REAL
		std::unique_ptr<ros::CallbackQueue> m_callbackQueue;
#endif //ROS_REAL
	};
}

#endif // ROS_FOUND
#endif // !__ROSWRAPPER_H__
