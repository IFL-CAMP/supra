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

#ifndef __ROSINTERFACE_H__
#define __ROSINTERFACE_H__

#include <string>
#include <mutex>
#include <ValueRangeDictionary.h>
#include <ConfigurationDictionary.h>
#include <AbstractInput.h>

#include <utilities/utility.h>
#include <utilities/RosWrapper.h>

#include <supra_msgs/parameter.h>
#include <supra_msgs/get_nodes.h>
#include <supra_msgs/get_node_parameters.h>
#include <supra_msgs/set_node_parameter.h>
#include <supra_msgs/freeze.h>
#include <supra_msgs/sequence.h>

namespace supra
{
	using std::to_string;

	//char* stringToNewCstr(std::string org);

	class ValueRangeDictionary;

#ifdef ROS_ROSSERIAL
	/// The return type of a service callback function. Differs between ros and rosserial_windows.
	typedef void ServiceReturnType;
#define ROSSERIAL_REQUESTTYPECONST const
#else
	/// The return type of a service callback function. Differs between ros and rosserial_windows.
	typedef bool ServiceReturnType;
#define ROSSERIAL_REQUESTTYPECONST
#endif

	/// The ros interface. This class maintains the services neccessary to provide an interface to 
	/// the nodes and their parameters. The interface works on linux with native ros and on 
	/// windows using rosserial_windows
	class RosInterface
	{
	public:
		/// The main method of the ros interface. Once called it returns only on shutdown of the 
		/// rosnode or complete stop of the compute graph
		static void mainLoop(std::string masterHost);

	private:
		static ServiceReturnType get_nodesCallback(ROSSERIAL_REQUESTTYPECONST supra_msgs::get_nodes::Request & req, supra_msgs::get_nodes::Response& res);
		static ServiceReturnType get_node_parametersCallback(ROSSERIAL_REQUESTTYPECONST supra_msgs::get_node_parameters::Request & req, supra_msgs::get_node_parameters::Response & res);
		static ServiceReturnType set_node_parameterCallback(ROSSERIAL_REQUESTTYPECONST supra_msgs::set_node_parameter::Request & req, supra_msgs::set_node_parameter::Response & res);
		static ServiceReturnType sequenceCallback(ROSSERIAL_REQUESTTYPECONST supra_msgs::sequence::Request & req, supra_msgs::sequence::Response& res);
		static void freezeCallback(const supra_msgs::freeze & freezeMsg);

		static std::string getParameterTypeString(const ValueRangeDictionary * ranges, std::string paramName);
		static std::string getParameterValueString(const ConfigurationDictionary * config, const ValueRangeDictionary * ranges, std::string paramName);

		static std::mutex m_accessMutex;

		static void fillParameterMessage(const ConfigurationDictionary* confDict, const ValueRangeDictionary * ranges, std::string paramName, supra_msgs::parameter* pParam);
		static bool convertAndSetParameter(std::string inputID, std::string paramName, std::string newValue);
		static bool sequence(bool active);
		static void freeze(bool freezeActive);

		template <typename ValueType>
		static void fillParameterMessage(const ConfigurationDictionary* confDict, const ValueRangeDictionary * ranges, std::string paramName, supra_msgs::parameter* pParam)
		{
			auto range = ranges->get<ValueType>(paramName);
			if (range->isUnrestricted())
			{
				pParam->rangeType = supra_msgs::parameter::rangeTypeUnrestricted;
			}
			else if (range->isContinuous())
			{
				pParam->rangeType = supra_msgs::parameter::rangeTypeContinuous;

				pParam->rangeStart = stringToNewCstr(to_string(range->getContinuous().first));
				pParam->rangeEnd = stringToNewCstr(to_string(range->getContinuous().second));
			}
			else
			{
				pParam->rangeType = supra_msgs::parameter::rangeTypeDiscrete;
				auto valueRange = range->getDiscrete();

				ROSSERIAL_MSG_SETUP_ARRAY(pParam->discreteValues, valueRange.size());
				for (size_t i = 0; i < valueRange.size(); i++)
				{
					pParam->discreteValues[i] = ROSSERIAL_MSG_STRING(to_string(valueRange[i]));
				}
			}
		}

		static bool m_sequenceActive;
	};
}

#endif //!__ROSINTERFACE_H__
