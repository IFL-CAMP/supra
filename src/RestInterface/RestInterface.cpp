/////// BASED ON: https://github.com/ivanmejiarocha/micro-service
/////// AND SUPRA
//
//  Created by Ivan Mejia on 12/24/16.
//
// MIT License
//
// Copyright (c) 2016 ivmeroLabs.
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <utilities/Logging.h>
#include <utilities/utility.h>

#include <SupraManager.h>


#include <std_micro_service.hpp>
#include "RestInterface.h"

using namespace web;
using namespace http;

namespace supra
{
	using namespace std;
	using namespace logging;
	
	void RestInterface::initRestOpHandlers() {
		_listener.support(methods::GET, std::bind(&RestInterface::handleGet, this, std::placeholders::_1));
		_listener.support(methods::PUT, std::bind(&RestInterface::handlePut, this, std::placeholders::_1));
		_listener.support(methods::POST, std::bind(&RestInterface::handlePost, this, std::placeholders::_1));
		_listener.support(methods::DEL, std::bind(&RestInterface::handleDelete, this, std::placeholders::_1));
		_listener.support(methods::PATCH, std::bind(&RestInterface::handlePatch, this, std::placeholders::_1));
	}

	void RestInterface::handleGet(http_request message) {
		auto path = requestPath(message);
		if (!path.empty()) 
		{
			/*if (path[0] == "service" && path[1] == "test") {
				auto response = json::value::object();
				response["version"] = json::value::string("0.1.1");
				response["status"] = json::value::string("ready!");
				message.reply(status_codes::OK, response);
			}*/
			
			if (path[0] == "nodes") 
			{
				auto nodeType = path[1];
				if (path.size() == 1)
                {
                    // MARK: Get request to /nodes is enough to get all the nodes
                    auto nodeType = "all";
                    auto nodeIDs = get_nodes(nodeType);
                    json::value responseIDs = json::value::array();

                    for (size_t k = 0; k < nodeIDs.size(); k++)
                    {
                        responseIDs[k] = json::value::string(nodeIDs[k]);
                    }
                    auto response = json::value::object();
                    response["nodeIDs"] = responseIDs;
                    message.reply(status_codes::OK, response);
                }
				else if (path.size()>1)
                {
                    auto nodeType = path[1];
                    auto nodeIDs = get_nodes(nodeType);
                    json::value responseIDs = json::value::array();

                    for (size_t k = 0; k < nodeIDs.size(); k++) {
                        responseIDs[k] = json::value::string(nodeIDs[k]);
                    }

                    auto response = json::value::object();
                    response["nodeIDs"] = responseIDs;
                    message.reply(status_codes::OK, response);
                }
								

			}
			// MARK: Brought the get Parameters under the get requests.
			else if (path[0] == "parameters")
			{
				auto reqJsonTask = message.extract_json();
				reqJsonTask.wait();
				auto reqJson = reqJsonTask.get();
				std::cout << "got request for params: " << reqJson.serialize() << std::endl;
                if (reqJson.is_object())
                {
                    auto reqObj = reqJson.as_object();
                    if (reqObj.find("nodeID") != reqObj.end()) {
                        std::string nodeID = reqObj["nodeID"].as_string();
                        auto response = get_node_parameters(nodeID);
                        message.reply(status_codes::OK, response);
                    }
                }
                else {
                    message.reply(status_codes::NotFound);
                }

            }
		}
		else {
			message.reply(status_codes::NotFound);
		}
	}

	void RestInterface::handlePatch(http_request message) {
		message.reply(status_codes::NotImplemented, responseNotImpl(methods::PATCH));
	}

	void RestInterface::handlePut(http_request message) {
		message.reply(status_codes::NotImplemented, responseNotImpl(methods::PUT));
	}

	void RestInterface::handlePost(http_request message) {
		auto path = requestPath(message);
		if (!path.empty()) 
		{
			/*if (path[0] == "service" && path[1] == "test") {
				auto response = json::value::object();
				response["version"] = json::value::string("0.1.1");
				response["status"] = json::value::string("ready!");
				message.reply(status_codes::OK, response);
			}*/

			if (path[0] == "parameters")
			{
				auto reqJsonTask = message.extract_json();
				reqJsonTask.wait();
				auto reqJson = reqJsonTask.get();
				std::cout << "got request to set param: " << reqJson.serialize() << std::endl;
				
				if (reqJson.is_object())
				{
					auto reqObj = reqJson.as_object();
					if (reqObj.find("nodeID") != reqObj.end() &&
						reqObj.find("parameterID") != reqObj.end() &&
						reqObj.find("value") != reqObj.end())
					{
						std::string nodeID = reqObj["nodeID"].as_string();
						std::string parameterID = reqObj["parameterID"].as_string();
						std::string parameterValue = reqObj["value"].as_string();
						
						std::cout << "got request to set parameter for '" << nodeID << "': '" << parameterID << "' = '" << parameterValue << "'" << std::endl;
						
						auto response = set_node_parameter(nodeID, parameterID, parameterValue);
						message.reply(status_codes::OK, response);
					}
				}
				else
				{
					message.reply(status_codes::NotFound);
				}
			}
			else {
				message.reply(status_codes::NotFound);
			}
		}
		else {
			message.reply(status_codes::NotFound);
		}
		//message.reply(status_codes::NotImplemented, responseNotImpl(methods::POST));
	}

	void RestInterface::handleDelete(http_request message) {    
		message.reply(status_codes::NotImplemented, responseNotImpl(methods::DEL));
	}

	void RestInterface::handleHead(http_request message) {
		message.reply(status_codes::NotImplemented, responseNotImpl(methods::HEAD));
	}

	void RestInterface::handleOptions(http_request message) {
		message.reply(status_codes::NotImplemented, responseNotImpl(methods::OPTIONS));
	}

	void RestInterface::handleTrace(http_request message) {
		message.reply(status_codes::NotImplemented, responseNotImpl(methods::TRCE));
	}

	void RestInterface::handleConnect(http_request message) {
		message.reply(status_codes::NotImplemented, responseNotImpl(methods::CONNECT));
	}

	void RestInterface::handleMerge(http_request message) {
		message.reply(status_codes::NotImplemented, responseNotImpl(methods::MERGE));
	}

	json::value RestInterface::responseNotImpl(const http::method & method) {
		auto response = json::value::object();
		response["serviceName"] = json::value::string("C++ Mircroservice Sample");
		response["http_method"] = json::value::string(method);
		return response ;
	}

	
	
	
	std::vector<std::string> RestInterface::get_nodes(const std::string& nodeType)
	{
		vector<string> nodeIDs;

		if (nodeType == "input")
		{
			log_always("Input nodes:");
			nodeIDs = SupraManager::Get()->getInputDeviceIDs();
		}
		if (nodeType == "output")
		{
			log_always("Output nodes:");
			nodeIDs = SupraManager::Get()->getOutputDeviceIDs();
		}
		if (nodeType == "all")
		{
			log_always("Nodes:");
			nodeIDs = SupraManager::Get()->getNodeIDs();
		}

		return nodeIDs;
	};
	
	json::value RestInterface::get_node_parameters(const std::string& nodeID)
	{
		json::value ret = json::value::object();
		auto node = SupraManager::Get()->getNode(nodeID);
		if(node != nullptr)
		{
			auto rangeDict = node->getValueRangeDictionary();
			auto confDict = node->getConfigurationDictionary();
			auto keys = rangeDict->getKeys();
			uint32_t numParams = static_cast<uint32_t>(keys.size());

			for (int i = 0; i < numParams; i++)
			{
				ret[keys[i]] = fillParameterMessage(confDict, rangeDict, keys[i]);
			}
		}
		return ret;
	}

	json::value RestInterface::set_node_parameter(const std::string& nodeID, const std::string& parameterID, const std::string& parameterValue)
	{
		json::value res = json::value::boolean(convertAndSetParameter(nodeID, parameterID, parameterValue));
		return res;
	}

	bool RestInterface::sequence(bool active)
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

	void RestInterface::freeze(bool freezeActive)
	{
		if(freezeActive)
		{
			SupraManager::Get()->freezeInputs();
			log_info("RestInterface: Freezing inputs");
		}
		else if(!freezeActive)
		{
			SupraManager::Get()->unfreezeInputs();
			log_info("RestInterface: Unfreezing inputs");
		}
	}
	
	json::value RestInterface::fillParameterMessage(const ConfigurationDictionary* confDict, const ValueRangeDictionary * rangeDict, std::string paramName)
	{
		json::value ret = json::value::object();
		if (rangeDict->hasKey(paramName))
		{
			ret["parameterId"] = json::value::string(paramName);
			ret["type"] = json::value::string(getParameterTypeString(rangeDict, paramName));
			ret["value"] = json::value::string(getParameterValueString(confDict, rangeDict, paramName));

			switch (rangeDict->getType(paramName))
			{
			case TypeBool:
				fillParameterMessage<bool>(confDict, rangeDict, paramName, ret);
				break;
			case TypeInt8:
				fillParameterMessage<int8_t>(confDict, rangeDict, paramName, ret);
				break;
			case TypeUint8:
				fillParameterMessage<uint8_t>(confDict, rangeDict, paramName, ret);
				break;
			case TypeInt16:
				fillParameterMessage<int16_t>(confDict, rangeDict, paramName, ret);
				break;
			case TypeUint16:
				fillParameterMessage<uint16_t>(confDict, rangeDict, paramName, ret);
				break;
			case TypeInt32:
				fillParameterMessage<int32_t>(confDict, rangeDict, paramName, ret);
				break;
			case TypeUint32:
				fillParameterMessage<uint32_t>(confDict, rangeDict, paramName, ret);
				break;
			case TypeInt64:
				fillParameterMessage<int64_t>(confDict, rangeDict, paramName, ret);
				break;
			case TypeUint64:
				fillParameterMessage<uint64_t>(confDict, rangeDict, paramName, ret);
				break;
			case TypeFloat:
				fillParameterMessage<float>(confDict, rangeDict, paramName, ret);
				break;
			case TypeDouble:
				fillParameterMessage<double>(confDict, rangeDict, paramName, ret);
				break;
			case TypeString:
				fillParameterMessage<string>(confDict, rangeDict, paramName, ret);
				break;
			case TypeDataType:
				fillParameterMessage<DataType>(confDict, rangeDict, paramName, ret);
				break;
			case TypeUnknown:
			default:
				log_warn("Cannot interpret parameter value range of '", paramName, "' because its type is unknown.");
				break;
			}
		}
		return ret;
	}
	
	bool RestInterface::convertAndSetParameter(std::string inputID, std::string paramName, std::string newValue)
	{
		bool wasSuccess = false;
		auto node = SupraManager::Get()->getNode(inputID);
		if (node != nullptr)
		{
			auto ranges = node->getValueRangeDictionary();

			stringstream sstream;
			sstream.str(newValue);
			if (ranges->hasKey(paramName))
			{
				switch (ranges->getType(paramName))
				{
				case TypeBool:
					bool valBool;
					sstream >> valBool;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<bool>(paramName, valBool);
					break;
				case TypeInt8:
					int8_t valInt8;
					sstream >> valInt8;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<int8_t>(paramName, valInt8);
					break;
				case TypeUint8:
					uint8_t valUint8;
					sstream >> valUint8;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<uint8_t>(paramName, valUint8);
					break;
				case TypeInt16:
					int16_t valInt16;
					sstream >> valInt16;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<int16_t>(paramName, valInt16);
					break;
				case TypeUint16:
					uint16_t valUint16;
					sstream >> valUint16;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<uint16_t>(paramName, valUint16);
					break;
				case TypeInt32:
					int32_t valInt32;
					sstream >> valInt32;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<int32_t>(paramName, valInt32);
					break;
				case TypeUint32:
					uint32_t valUint32;
					sstream >> valUint32;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<uint32_t>(paramName, valUint32);
					break;
				case TypeInt64:
					int64_t valInt64;
					sstream >> valInt64;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<int64_t>(paramName, valInt32);
					break;
				case TypeUint64:
					uint64_t valUint64;
					sstream >> valUint64;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<uint64_t>(paramName, valUint32);
					break;
				case TypeFloat:
					float valFloat;
					sstream >> valFloat;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<float>(paramName, valFloat);
					break;
				case TypeDouble:
					double valDouble;
					sstream >> valDouble;
					if (!sstream.fail())
						wasSuccess = node->changeConfig<double>(paramName, valDouble);
					break;
				case TypeString:
					wasSuccess = node->changeConfig<string>(paramName, newValue);
					break;
				case TypeDataType:
					DataType valDataType;
					valDataType = from_string<DataType>(newValue);
					wasSuccess = node->changeConfig<DataType>(paramName, valDataType);
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
		}
		return wasSuccess;
	}

	std::string RestInterface::getParameterTypeString(const ValueRangeDictionary * ranges, std::string paramName)
	{
		string retVal = "";
		if (ranges->hasKey(paramName))
		{
			switch (ranges->getType(paramName))
			{
			case TypeBool:
				retVal = "TypeBool";
				break;
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
			case TypeDataType:
				retVal = "TypeDataType";
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

	string RestInterface::getParameterValueString(const ConfigurationDictionary * config, const ValueRangeDictionary * ranges, std::string paramName)
	{
		string retVal = "";
		if (ranges->hasKey(paramName))
		{
			switch (ranges->getType(paramName))
			{
			case TypeBool:
				retVal = stringify(config->get<bool>(paramName, 0));
				break;
			case TypeInt8:
				retVal = stringify(config->get<int8_t>(paramName, 0));
				break;
			case TypeUint8:
				retVal = stringify(config->get<uint8_t>(paramName, 0));
				break;
			case TypeInt16:
				retVal = stringify(config->get<int16_t>(paramName, 0));
				break;
			case TypeUint16:
				retVal = stringify(config->get<uint16_t>(paramName, 0));
				break;
			case TypeInt32:
				retVal = stringify(config->get<int32_t>(paramName, 0));
				break;
			case TypeUint32:
				retVal = stringify(config->get<uint32_t>(paramName, 0));
				break;
			case TypeInt64:
				retVal = stringify(config->get<int64_t>(paramName, 0));
				break;
			case TypeUint64:
				retVal = stringify(config->get<uint64_t>(paramName, 0));
				break;
			case TypeFloat:
				retVal = stringify(config->get<float>(paramName, 0.0f));
				break;
			case TypeDouble:
				retVal = stringify(config->get<double>(paramName, 0.0));
				break;
			case TypeString:
				retVal = (config->get<string>(paramName, ""));
				break;
			case TypeDataType:
				retVal = stringify(config->get<DataType>(paramName, TypeUnknown));
				break;
			case TypeUnknown:
			default:
				log_warn("Cannot get parameter '", paramName, "' because its type is unknown.");
				break;
			}
		}
		return retVal;
	}
	
}