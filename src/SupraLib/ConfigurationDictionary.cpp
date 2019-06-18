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

#include <string>

#include "utilities/Logging.h"
#include "utilities/utility.h"
#include "ConfigurationDictionary.h"

using namespace std;

namespace supra
{
	using namespace logging;

	ConfigurationDictionary::ConfigurationDictionary(const tinyxml2::XMLElement * parentXmlElement)
		:p_valueRangeDictionary(nullptr)
	{
		const tinyxml2::XMLElement* parameterElement = parentXmlElement->FirstChildElement("param");
		while (parameterElement)
		{
			string paramName = parameterElement->Attribute("name");
			string paramType = parameterElement->Attribute("type");

			if (paramType == "uint32_t")
			{
				uint32_t val = 0;
				if (parameterElement->QueryUnsignedText(&val) != tinyxml2::XML_SUCCESS)
				{
					log_error("Error parsing parameter '", paramName, "' of type '", paramType, "'.");
				}
				else {
					set<uint32_t>(paramName, val);
				}
			}
			else if (paramType == "int")
			{
				int val = 0;
				if (parameterElement->QueryIntText(&val) != tinyxml2::XML_SUCCESS)
				{
					log_error("Error parsing parameter '", paramName, "' of type '", paramType, "'.");
				}
				else {
					set<int>(paramName, val);
				}
			}
			else if (paramType == "size_t")
			{
				int64_t val = 0;
				if (parameterElement->QueryInt64Text(&val) != tinyxml2::XML_SUCCESS)
				{
					log_error("Error parsing parameter '", paramName, "' of type '", paramType, "'.");
				}
				else {
					set<size_t>(paramName, (size_t)val);
				}
			}
			else if (paramType == "bool")
			{
				bool val = false;
				if (parameterElement->QueryBoolText(&val) != tinyxml2::XML_SUCCESS)
				{
					log_error("Error parsing parameter '", paramName, "' of type '", paramType, "'.");
				}
				else {
					set<bool>(paramName, val);
				}
			}
			else if (paramType == "double")
			{
				double val = 0.0;
				if (parameterElement->QueryDoubleText(&val) != tinyxml2::XML_SUCCESS)
				{
					log_error("Error parsing parameter '", paramName, "' of type '", paramType, "'.");
				}
				else {
					set<double>(paramName, val);
				}
			}
			else if (paramType == "string")
			{
				string val = trim(parameterElement->GetText());
				set<string>(paramName, val);
			}
			else if (paramType == "DataType")
			{
				string val = trim(parameterElement->GetText());
				bool success;
				DataType valTyped = DataTypeFromString(val, &success);
				if (!success)
				{
					log_error("Error parsing parameter '", paramName, "' of type '", paramType, "'.");
				}
				else {
					set<DataType>(paramName, valTyped);
				}
			}
			else
			{
				log_error("Unknown parameter type for '", paramName, "'. Type '", paramType, "'.");
			}

			parameterElement = parameterElement->NextSiblingElement("param");
		}
	}

	void ConfigurationDictionary::toXml(tinyxml2::XMLElement * parentXmlElement) const
	{
		if (p_valueRangeDictionary != nullptr)
		{
			auto doc = parentXmlElement->GetDocument();

			for (auto entryPair : m_mapEntries)
			{
				auto entryName = entryPair.first;
				auto entry = entryPair.second;
				auto entryType = p_valueRangeDictionary->getType(entryName);
				auto entryTypeString = DataTypeToString(entryType);
				
				auto parameterElement = doc->NewElement("param");
				parentXmlElement->InsertEndChild(parameterElement);
				parameterElement->SetAttribute("name", entryName.c_str());
				parameterElement->SetAttribute("type", entryTypeString.c_str());

				std::string entryValueString;

				switch (entryType)
				{
				case TypeBool:
					entryValueString = stringify(get<bool>(entryName));
					break;
				case TypeInt8:
					entryValueString = stringify(get<int8_t>(entryName));
					break;
				case TypeUint8:
					entryValueString = stringify(get<uint8_t>(entryName));
					break;
				case TypeInt16:
					entryValueString = stringify(get<int16_t>(entryName));
					break;
				case TypeUint16:
					entryValueString = stringify(get<uint16_t>(entryName));
					break;
				case TypeInt32:
					entryValueString = stringify(get<int32_t>(entryName));
					break;
				case TypeUint32:
					entryValueString = stringify(get<uint32_t>(entryName));
					break;
				case TypeInt64:
					entryValueString = stringify(get<int64_t>(entryName));
					break;
				case TypeUint64:
					entryValueString = stringify(get<uint64_t>(entryName));
					break;
				case TypeFloat:
					entryValueString = stringify(get<float>(entryName));
					break;
				case TypeDouble:
					entryValueString = stringify(get<double>(entryName));
					break;
				case TypeString:
					entryValueString = get<std::string>(entryName);
					break;
				case TypeDataType:
					entryValueString = stringify(get<DataType>(entryName));
					break;
				case TypeUnknown:
				default:
					logging::log_error("Cannot save the value of configuration entry '", entryName, "' as its range type is unknown.");
				}

				parameterElement->SetText(entryValueString.c_str());
			}
		}
	}
}