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

#include "CommandlineInterface.h"

#include <iostream>
#include <memory>
#include <string>
#include <bitset>

#include <utilities/Logging.h>

#include <SupraManager.h>

namespace supra
{
	using namespace std;
	using namespace logging;

	void CommandlineInterface::mainMenu()
	{
		shared_ptr<SupraManager> manager = SupraManager::Get();

		logging::SeverityMask currentLogLevel = logging::error | logging::warning;

		bool continueLoop = true;
		while (continueLoop) {
			log_always("SUPRA_CMD (h for help): ");
			char inputChar;
			cin >> inputChar;

			unsigned int nodeNum = 0;
			logging::Severity newLogLevelEn = logging::log;

			switch (inputChar)
			{
				//help
			case 'h':
				log_always("Usage:\n",
					"h - help\n",
					"q - quit\n",
					"li - list input nodes\n",
					"lo - list output nodes\n",
					"lp - list all nodes\n",
					"i<num> - edit input node #num\n",
					"o<num> - edit output node #num\n",
					"n<num> - edit node #num\n",
					"k - set log level");
				break;
				//Quit the program
			case 'q':
				continueLoop = false;
				break;
			case 'l':
				//list one kind of node
				cin >> inputChar;
				switch (inputChar)
				{
				case 'i':
					log_always("Input nodes:");
					for (string inID : manager->getInputDeviceIDs())
					{
						log_always(inID);
					}
					break;
				case 'o':
					log_always("Output nodes:");
					for (string outID : manager->getOutputDeviceIDs())
					{
						log_always(outID);
					}
					break;
				case 'p':
					log_always("Nodes:");
					for (string nodeID : manager->getNodeIDs())
					{
						log_always(nodeID);
					}
					break;
				default:
					break;
				}
				break;
			case 'i':
				cin >> nodeNum;
				editInputNode(nodeNum);
				break;
			case 'o':
				cin >> nodeNum;
				editOutputNode(nodeNum);
				break;
			case 'p':
				cin >> nodeNum;
				editNode(nodeNum);
				break;
			case 'k':
				log_always("Log level: ", std::bitset<6>(currentLogLevel), "\n",
					"  Available:\n",
					"    ", (unsigned int)logging::log, " = log\n",
					"    ", (unsigned int)logging::info, " = info\n",
					"    ", (unsigned int)logging::warning, " = warning\n",
					"    ", (unsigned int)logging::error, " = error\n",
					"    ", (unsigned int)logging::param, " = parameter\n",
					"    ", (unsigned int)logging::external, " = externals\n");
				unsigned int newLogLevel;
				cin >> newLogLevel;
				newLogLevelEn = static_cast<logging::Severity>(newLogLevel);
				if (newLogLevelEn == logging::log ||
					newLogLevelEn == logging::info ||
					newLogLevelEn == logging::warning ||
					newLogLevelEn == logging::error ||
					newLogLevelEn == logging::param ||
					newLogLevelEn == logging::external)
				{
					currentLogLevel = currentLogLevel ^ newLogLevelEn;
					logging::Base::setLogLevel(currentLogLevel);
				}
			default:
				break;
			}
		}
	}

	void CommandlineInterface::editInputNode(unsigned int nodeNum)
	{
		string inputID = SupraManager::Get()->getInputDeviceIDs()[nodeNum];
		auto input = SupraManager::Get()->getInputDevice(inputID);
		bool continueLoop = true;
		while (continueLoop) {
			log_always("Input device '", inputID, "': ");

			auto rangeDict = input->getValueRangeDictionary();
			auto confDict = input->getConfigurationDictionary();
			for (string param : rangeDict->getKeys())
			{
				log_always("  ", param, " = ",
					getParameterValueString(confDict, rangeDict, param),
					" (",
					getParameterTypeString(rangeDict, param), ")");
			}

			string paramToChange;
			cin >> paramToChange;
			if (paramToChange == "quit")
			{
				continueLoop = false;
			}
			else {
				if (rangeDict->hasKey(paramToChange))
				{
					log_always("  Valid values:   ('{}' = discrete set, '[]' = inclusive range)\n",
						getParameterRangeString(rangeDict, paramToChange));
					readAndSetParameter(input, paramToChange);
				}
			}
		}
	}

	void CommandlineInterface::editOutputNode(unsigned int nodeNum)
	{
	}

	void CommandlineInterface::editNode(unsigned int nodeNum)
	{
	}

	string CommandlineInterface::getParameterRangeString(const ValueRangeDictionary * ranges, std::string paramName)
	{
		string retVal = "";
		if (ranges->hasKey(paramName))
		{
			switch (ranges->getType(paramName))
			{
			case TypeInt8:
				retVal = getParameterRangeStringTemplated<int8_t>(ranges, paramName);
				break;
			case TypeUint8:
				retVal = getParameterRangeStringTemplated<uint8_t>(ranges, paramName);
				break;
			case TypeInt16:
				retVal = getParameterRangeStringTemplated<int16_t>(ranges, paramName);
				break;
			case TypeUint16:
				retVal = getParameterRangeStringTemplated<uint16_t>(ranges, paramName);
				break;
			case TypeInt32:
				retVal = getParameterRangeStringTemplated<int32_t>(ranges, paramName);
				break;
			case TypeUint32:
				retVal = getParameterRangeStringTemplated<uint32_t>(ranges, paramName);
				break;
			case TypeInt64:
				retVal = getParameterRangeStringTemplated<int64_t>(ranges, paramName);
				break;
			case TypeUint64:
				retVal = getParameterRangeStringTemplated<uint64_t>(ranges, paramName);
				break;
			case TypeFloat:
				retVal = getParameterRangeStringTemplated<float>(ranges, paramName);
				break;
			case TypeDouble:
				retVal = getParameterRangeStringTemplated<double>(ranges, paramName);
				break;
			case TypeString:
				retVal = getParameterRangeStringTemplated<string>(ranges, paramName);
				break;
			case TypeValueUnknown:
			default:
				log_warn("Cannot interpret parameter value range of '", paramName, "' because its type is unknown.");
				break;
			}
		}
		return retVal;
	}

	void CommandlineInterface::readAndSetParameter(std::shared_ptr<AbstractInput<RecordObject> > inputNode, std::string paramName)
	{
		auto ranges = inputNode->getValueRangeDictionary();
		if (ranges->hasKey(paramName))
		{
			switch (ranges->getType(paramName))
			{
			case TypeInt8:
				int8_t valInt8;
				cin >> valInt8;
				inputNode->changeConfig<int8_t>(paramName, valInt8);
				break;
			case TypeUint8:
				uint8_t valUint8;
				cin >> valUint8;
				inputNode->changeConfig<uint8_t>(paramName, valUint8);
				break;
			case TypeInt16:
				int16_t valInt16;
				cin >> valInt16;
				inputNode->changeConfig<int16_t>(paramName, valInt16);
				break;
			case TypeUint16:
				uint16_t valUint16;
				cin >> valUint16;
				inputNode->changeConfig<uint16_t>(paramName, valUint16);
				break;
			case TypeInt32:
				int32_t valInt32;
				cin >> valInt32;
				inputNode->changeConfig<int32_t>(paramName, valInt32);
				break;
			case TypeUint32:
				uint32_t valUint32;
				cin >> valUint32;
				inputNode->changeConfig<uint32_t>(paramName, valUint32);
				break;
			case TypeInt64:
				int64_t valInt64;
				cin >> valInt64;
				inputNode->changeConfig<int64_t>(paramName, valInt32);
				break;
			case TypeUint64:
				uint64_t valUint64;
				cin >> valUint64;
				inputNode->changeConfig<uint64_t>(paramName, valUint32);
				break;
			case TypeFloat:
				float valFloat;
				cin >> valFloat;
				inputNode->changeConfig<float>(paramName, valFloat);
				break;
			case TypeDouble:
				double valDouble;
				cin >> valDouble;
				inputNode->changeConfig<double>(paramName, valDouble);
				break;
			case TypeString:
			{
				string valString;
				cin >> valString;
				inputNode->changeConfig<string>(paramName, valString);
			}
			break;
			case TypeValueUnknown:
			default:
				log_warn("Cannot set parameter '", paramName, "' because its type is unknown.");
				break;
			}

			if (!cin.good())
			{
				log_always("Could not parse the input as the required datatype. Try again.");
				cin.clear();
				cin.ignore();
			}
		}
	}

	std::string CommandlineInterface::getParameterTypeString(const ValueRangeDictionary * ranges, std::string paramName)
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
			case TypeValueUnknown:
			default:
				retVal = "TypeValueUnknown";
				log_warn("Encountered unknown parameter type for '", paramName, "'.");
				break;
			}
		}
		return retVal;
	}

	string CommandlineInterface::getParameterValueString(const ConfigurationDictionary * config, const ValueRangeDictionary * ranges, std::string paramName)
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
			case TypeValueUnknown:
			default:
				log_warn("Cannot get parameter '", paramName, "' because its type is unknown.");
				break;
			}
		}
		return retVal;
	}
}
