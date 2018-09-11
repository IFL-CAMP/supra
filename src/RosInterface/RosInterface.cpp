// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl
//      Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "RosInterface.h"

//#include <utilities/RosWrapper.h>
//#include <supra_msgs/get_nodes.h>
//#include <supra_msgs/get_node_parameters.h>
//#include <supra_msgs/set_node_parameter.h>
//#include <supra_msgs/freeze.h>
//#include <supra_msgs/sequence.h>

#include <iostream>
#include <memory>
#include <string>

#include <utilities/Logging.h>
#include <utilities/utility.h>

#include <SupraManager.h>

namespace supra
{
	using namespace std;
	using namespace logging;

#ifdef ROS_ROSSERIAL
	class GetNodesServer : public ros::ServiceServer<supra_msgs::get_nodes::Request, supra_msgs::get_nodes::Response>
	{
	public:
		GetNodesServer(const char* topic_name, CallbackT cb)
			:ros::ServiceServer<supra_msgs::get_nodes::Request, supra_msgs::get_nodes::Response>(topic_name, nullptr)
			, cb_allocating(cb)
		{};

		virtual void callback(unsigned char *data) {
			req.deserialize(data);
			cb_allocating(req, resp);
			pub.publish(&resp);
			for (unsigned int i = 0; i < resp.ids_length; i++)
			{
				delete[] resp.ids[i];
			}
			delete[] resp.ids;
		};
	protected:
		CallbackT cb_allocating;

	};

	class GetNodeParametersServer : public ros::ServiceServer<supra_msgs::get_node_parameters::Request, supra_msgs::get_node_parameters::Response>
	{
	public:
		GetNodeParametersServer(const char* topic_name, CallbackT cb)
			:ros::ServiceServer<supra_msgs::get_node_parameters::Request, supra_msgs::get_node_parameters::Response>(topic_name, nullptr)
			, cb_allocating(cb)
		{};

		virtual void callback(unsigned char *data) {
			req.deserialize(data);
			cb_allocating(req, resp);
			pub.publish(&resp);
			for (unsigned int i = 0; i < resp.parameters_length; i++)
			{
				delete[] resp.parameters[i].parameterId;
				//delete[] resp.parameters[i].displayName;
				delete[] resp.parameters[i].type;
				delete[] resp.parameters[i].value;
				switch (resp.parameters[i].rangeType)
				{
				case supra_msgs::parameter::rangeTypeUnrestricted:
					break;
				case supra_msgs::parameter::rangeTypeContinuous:
					delete[] resp.parameters[i].rangeStart;
					delete[] resp.parameters[i].rangeEnd;
					break;
				case supra_msgs::parameter::rangeTypeDiscrete:
					for (unsigned int j = 0; j < resp.parameters[i].discreteValues_length; j++)
					{
						delete[] resp.parameters[i].discreteValues[j];
					}
					delete[] resp.parameters[i].discreteValues;
					break;
				}
			}
			delete[] resp.parameters;
		};
	protected:
		CallbackT cb_allocating;
	};
#endif

	bool supra::RosInterface::m_sequenceActive = false;

	void RosInterface::mainLoop(string masterHost)
	{
		logging::log_info("RosInterface: Connecting to master '", masterHost, "'");
		RosWrapper wr(masterHost);
		m_sequenceActive = false;

#ifdef ROS_ROSSERIAL
		GetNodesServer servGetNodes(supra_msgs::GET_NODES, RosInterface::get_nodesCallback);
		bool getNodesAdvertisement = wr.getNodeHandle()->advertiseService(servGetNodes);
		logging::log_error_if(!getNodesAdvertisement, "Error advertising service ", "get_nodes");

		GetNodeParametersServer servGetNodeParameters
		(supra_msgs::GET_NODE_PARAMETERS, (RosInterface::get_node_parametersCallback));
		bool getNodeParametersAdvertisement = wr.getNodeHandle()->advertiseService(servGetNodeParameters);
		logging::log_error_if(!getNodeParametersAdvertisement, "Error advertising service ", "get_node_parameters");

		ros::ServiceServer<supra_msgs::set_node_parameterRequest, supra_msgs::set_node_parameterResponse> servSetNodeParameter
		(supra_msgs::SET_NODE_PARAMETER, (RosInterface::set_node_parameterCallback));
		bool setNodeParameterAdvertisement = wr.getNodeHandle()->advertiseService(servSetNodeParameter);
		logging::log_error_if(!setNodeParameterAdvertisement, "Error advertising service ", "set_node_parameter");

		ros::ServiceServer<supra_msgs::sequenceRequest, supra_msgs::sequenceResponse> servSequence
		(supra_msgs::SEQUENCE, (RosInterface::sequenceCallback));
		bool sequenceAdvertisement = wr.getNodeHandle()->advertiseService(servSequence);
		logging::log_error_if(!sequenceAdvertisement, "Error advertising service ", "sequence");
		wr.subscribe<supra_msgs::freeze, RosInterface>("supra_freeze", RosInterface::freezeCallback);
#else
		auto serviceGetNodes = wr.getNodeHandle()->advertiseService("get_nodes", RosInterface::get_nodesCallback);
		auto serviceGetNodeParameters = wr.getNodeHandle()->advertiseService("get_node_parameters", RosInterface::get_node_parametersCallback);
		auto serviceSetNodeParameters = wr.getNodeHandle()->advertiseService("set_node_parameter", RosInterface::set_node_parameterCallback);
		auto serviceSequence = wr.getNodeHandle()->advertiseService("sequence", RosInterface::sequenceCallback);
		wr.subscribe<supra_msgs::freeze>("supra_freeze", RosInterface::freezeCallback);
#endif

		wr.spinEndless();

		logging::log_log("RosInterface ended");
	}

	ServiceReturnType RosInterface::get_nodesCallback(ROSSERIAL_REQUESTTYPECONST supra_msgs::get_nodes::Request & req, supra_msgs::get_nodes::Response& res)
	{
		vector<string> nodeIDs;

		switch (req.type)
		{
		case supra_msgs::get_nodesRequest::typeInput:
			log_always("Input nodes:");
			nodeIDs = SupraManager::Get()->getInputDeviceIDs();
			break;
		case supra_msgs::get_nodesRequest::typeOutput:
			log_always("Output nodes:");
			nodeIDs = SupraManager::Get()->getOutputDeviceIDs();
			break;
		case supra_msgs::get_nodesRequest::typeAll:
		default:
			log_always("Nodes:");
			nodeIDs = SupraManager::Get()->getNodeIDs();
			break;
		}
#ifdef ROS_ROSSERIAL
		res.ids_length = static_cast<uint32_t>(nodeIDs.size());
		res.ids = new char*[res.ids_length];
		for (unsigned int i = 0; i < res.ids_length; i++)
		{
			res.ids[i] = stringToNewCstr(nodeIDs[i]);
		}
#else
		logging::log_error("native ros not implemented yet");
		return true;
#endif
	};

	ServiceReturnType RosInterface::get_node_parametersCallback(ROSSERIAL_REQUESTTYPECONST supra_msgs::get_node_parameters::Request & req, supra_msgs::get_node_parameters::Response & res)
	{
		std::string nodeID(req.nodeId);

		auto input = SupraManager::Get()->getInputDevice(nodeID);
		auto rangeDict = input->getValueRangeDictionary();
		auto confDict = input->getConfigurationDictionary();
		auto keys = rangeDict->getKeys();
		uint32_t numParams = static_cast<uint32_t>(keys.size());

		ROSSERIAL_MSG_SETUP_ARRAY(res.parameters, numParams);
		for (int i = 0; i < numParams; i++)
		{
			fillParameterMessage(confDict, rangeDict, keys[i], &res.parameters[i]);
		}
#ifdef ROS_REAL
		return true;
#endif
	}

	ServiceReturnType RosInterface::set_node_parameterCallback(ROSSERIAL_REQUESTTYPECONST supra_msgs::set_node_parameter::Request & req, supra_msgs::set_node_parameter::Response & res)
	{
		res.wasValid = convertAndSetParameter(req.nodeId, req.parameterId, req.value);
#ifdef ROS_REAL
		return res.wasValid;
#endif
	}

	ServiceReturnType RosInterface::sequenceCallback(ROSSERIAL_REQUESTTYPECONST supra_msgs::sequence::Request & req, supra_msgs::sequence::Response& res)
	{
		res.success = sequence(req.sequenceActive);
#ifdef ROS_REAL
		return res.success;
#endif
	}

	void RosInterface::freezeCallback(const supra_msgs::freeze & freezeMsg)
	{
		freeze(freezeMsg.freezeActive);
	}

	void RosInterface::fillParameterMessage(const ConfigurationDictionary* confDict, const ValueRangeDictionary * rangeDict, std::string paramName, supra_msgs::parameter* pParam)
	{
		if (rangeDict->hasKey(paramName))
		{
			pParam->parameterId = stringToNewCstr(paramName);
			pParam->type = stringToNewCstr(getParameterTypeString(rangeDict, paramName));
			pParam->value = stringToNewCstr(getParameterValueString(confDict, rangeDict, paramName));

			switch (rangeDict->getType(paramName))
			{
			case TypeInt8:
				fillParameterMessage<int8_t>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeUint8:
				fillParameterMessage<uint8_t>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeInt16:
				fillParameterMessage<int16_t>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeUint16:
				fillParameterMessage<uint16_t>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeInt32:
				fillParameterMessage<int32_t>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeUint32:
				fillParameterMessage<uint32_t>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeInt64:
				fillParameterMessage<int64_t>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeUint64:
				fillParameterMessage<uint64_t>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeFloat:
				fillParameterMessage<float>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeDouble:
				fillParameterMessage<double>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeString:
				fillParameterMessage<string>(confDict, rangeDict, paramName, pParam);
				break;
			case TypeUnknown:
			default:
				log_warn("Cannot interpret parameter value range of '", paramName, "' because its type is unknown.");
				break;
			}
		}
	}

	bool RosInterface::convertAndSetParameter(std::string inputID, std::string paramName, std::string newValue)
	{
		auto inputNode = SupraManager::Get()->getInputDevice(inputID);
		auto ranges = inputNode->getValueRangeDictionary();

		bool wasSuccess = false;

		stringstream sstream;
		sstream.str(newValue);
		if (ranges->hasKey(paramName))
		{
			switch (ranges->getType(paramName))
			{
			case TypeInt8:
				int8_t valInt8;
				sstream >> valInt8;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<int8_t>(paramName, valInt8);
				break;
			case TypeUint8:
				uint8_t valUint8;
				sstream >> valUint8;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<uint8_t>(paramName, valUint8);
				break;
			case TypeInt16:
				int16_t valInt16;
				sstream >> valInt16;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<int16_t>(paramName, valInt16);
				break;
			case TypeUint16:
				uint16_t valUint16;
				sstream >> valUint16;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<uint16_t>(paramName, valUint16);
				break;
			case TypeInt32:
				int32_t valInt32;
				sstream >> valInt32;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<int32_t>(paramName, valInt32);
				break;
			case TypeUint32:
				uint32_t valUint32;
				sstream >> valUint32;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<uint32_t>(paramName, valUint32);
				break;
			case TypeInt64:
				int64_t valInt64;
				sstream >> valInt64;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<int64_t>(paramName, valInt32);
				break;
			case TypeUint64:
				uint64_t valUint64;
				sstream >> valUint64;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<uint64_t>(paramName, valUint32);
				break;
			case TypeFloat:
				float valFloat;
				sstream >> valFloat;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<float>(paramName, valFloat);
				break;
			case TypeDouble:
				double valDouble;
				sstream >> valDouble;
				if (!sstream.fail())
					wasSuccess = inputNode->changeConfig<double>(paramName, valDouble);
				break;
			case TypeString:
				wasSuccess = inputNode->changeConfig<string>(paramName, newValue);
				break;
			case TypeUnknown:
			default:
				log_warn("Cannot set parameter '", paramName, "' because its type is unknown.");
				break;
			}

			if (sstream.fail())
			{
				log_warn("Could not parse the new parameter value as the required datatype.");
				wasSuccess = false;
			}
		}
		return wasSuccess;
	}

	std::string RosInterface::getParameterTypeString(const ValueRangeDictionary * ranges, std::string paramName)
	{
		string retVal = "";
		if (ranges->hasKey(paramName))
		{
			switch (ranges->getType(paramName))
			{
			case TypeInt8:
				retVal = "TypeInt8";
				break;
			case TypeUint8:
				retVal = "TypeUint8";
				break;
			case TypeInt16:
				retVal = "TypeInt16";
				break;
			case TypeUint16:
				retVal = "TypeUint16";
				break;
			case TypeInt32:
				retVal = "TypeInt32";
				break;
			case TypeUint32:
				retVal = "TypeUint32";
				break;
			case TypeInt64:
				retVal = "TypeInt64";
				break;
			case TypeUint64:
				retVal = "TypeUint64";
				break;
			case TypeFloat:
				retVal = "TypeFloat";
				break;
			case TypeDouble:
				retVal = "TypeDouble";
				break;
			case TypeString:
				retVal = "TypeString";
				break;
			case TypeUnknown:
			default:
				retVal = "TypeValueUnknown";
				log_warn("Encountered unknown parameter type for '", paramName, "'.");
				break;
			}
		}
		return retVal;
	}

	string RosInterface::getParameterValueString(const ConfigurationDictionary * config, const ValueRangeDictionary * ranges, std::string paramName)
	{
		string retVal = "";
		if (ranges->hasKey(paramName))
		{
			switch (ranges->getType(paramName))
			{
			case TypeInt8:
				retVal = to_string(config->get<int8_t>(paramName, 0));
				break;
			case TypeUint8:
				retVal = to_string(config->get<uint8_t>(paramName, 0));
				break;
			case TypeInt16:
				retVal = to_string(config->get<int16_t>(paramName, 0));
				break;
			case TypeUint16:
				retVal = to_string(config->get<uint16_t>(paramName, 0));
				break;
			case TypeInt32:
				retVal = to_string(config->get<int32_t>(paramName, 0));
				break;
			case TypeUint32:
				retVal = to_string(config->get<uint32_t>(paramName, 0));
				break;
			case TypeInt64:
				retVal = to_string(config->get<int64_t>(paramName, 0));
				break;
			case TypeUint64:
				retVal = to_string(config->get<uint64_t>(paramName, 0));
				break;
			case TypeFloat:
				retVal = to_string(config->get<float>(paramName, 0.0f));
				break;
			case TypeDouble:
				retVal = to_string(config->get<double>(paramName, 0.0));
				break;
			case TypeString:
				retVal = (config->get<string>(paramName, ""));
				break;
			case TypeUnknown:
			default:
				log_warn("Cannot get parameter '", paramName, "' because its type is unknown.");
				break;
			}
		}
		return retVal;
	}

	bool RosInterface::sequence(bool active)
	{
		bool ret = false;
		if(m_sequenceActive && !active)
		{
			SupraManager::Get()->stopOutputsSequence();
			m_sequenceActive = false;
			ret = true;
		}
		else if(!m_sequenceActive && active)
		{
			SupraManager::Get()->startOutputsSequence();
			m_sequenceActive = true;
			ret = true;
		}
		return ret;
	}

	void RosInterface::freeze(bool freezeActive)
	{
		if(freezeActive)
		{
			SupraManager::Get()->freezeInputs();
			log_info("RosInterface: Freezing inputs");
		}
		else if(!freezeActive)
		{
			SupraManager::Get()->unfreezeInputs();
			log_info("RosInterface: Unfreezing inputs");
		}
	}
}
