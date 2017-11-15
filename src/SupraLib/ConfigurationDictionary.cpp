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
			else
			{
				log_error("Unknown parameter type for '", paramName, "'. Type '", paramType, "'.");
			}

			parameterElement = parameterElement->NextSiblingElement("param");
		}
	}
}