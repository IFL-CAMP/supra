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

#pragma once 

#include <string>
#include <mutex>
#include <ValueRangeDictionary.h>
#include <ConfigurationDictionary.h>
#include <AbstractInput.h>

#include <utilities/utility.h>

#include <basic_controller.hpp>

using namespace cfx;

namespace supra
{
	using std::to_string;

	class ValueRangeDictionary;

	
	class RestInterface : public BasicController, Controller {
	public:
		RestInterface() : BasicController() {}
		~RestInterface() {}
		void handleGet(http_request message) override;
		void handlePut(http_request message) override;
		void handlePost(http_request message) override;
		void handlePatch(http_request message) override;
		void handleDelete(http_request message) override;
		void handleHead(http_request message) override;
		void handleOptions(http_request message) override;
		void handleTrace(http_request message) override;
		void handleConnect(http_request message) override;
		void handleMerge(http_request message) override;
		void initRestOpHandlers() override;    

	private:
		static json::value responseNotImpl(const http::method & method);

	private:
		std::vector<std::string> get_nodes(const std::string& nodeType);
		json::value get_node_parameters(const std::string& nodeID);
		json::value set_node_parameter(const std::string& nodeID, const std::string& parameterID, const std::string& parameterValue);

		std::string getParameterTypeString(const ValueRangeDictionary * ranges, std::string paramName);
		std::string getParameterValueString(const ConfigurationDictionary * config, const ValueRangeDictionary * ranges, std::string paramName);

		std::mutex m_accessMutex;

		json::value fillParameterMessage(const ConfigurationDictionary* confDict, const ValueRangeDictionary * ranges, std::string paramName);
		bool convertAndSetParameter(std::string inputID, std::string paramName, std::string newValue);
		bool sequence(bool active);
		void freeze(bool freezeActive);

		template <typename ValueType>
		void fillParameterMessage(const ConfigurationDictionary* confDict, const ValueRangeDictionary * ranges, std::string paramName, json::value& retObj)
		{
			auto range = ranges->get<ValueType>(paramName);
			if (range->isUnrestricted())
			{
				retObj["rangeType"] = json::value::string("unrestricted");
			}
			else if (range->isContinuous())
			{
				retObj["rangeType"] = json::value::string("continuous");

				retObj["rangeStart"] = json::value::string(stringify(range->getContinuous().first));
				retObj["rangeEnd"] = json::value::string(stringify(range->getContinuous().second));
			}
			else
			{
				retObj["rangeType"] = json::value::string("discrete");
				auto valueRange = range->getDiscrete();

				retObj["discreteValues"] = json::value::array();
				for (size_t i = 0; i < valueRange.size(); i++)
				{
					retObj["discreteValues"][i] = json::value::string(stringify(valueRange[i]));
				}
			}
		}

		bool m_sequenceActive;
	};
}
