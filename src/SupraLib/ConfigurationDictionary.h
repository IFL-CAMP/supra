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

#ifndef __CONFIGURATIONDICTIONARY_H__
#define __CONFIGURATIONDICTIONARY_H__

#include <string>
#include <memory>
#include <tuple>
#include <map>
#include "utilities/tinyxml2/tinyxml2.h"
#include "utilities/TemplateTypeDefault.h"
#include <utilities/Logging.h>

#include "ValueRangeDictionary.h"

namespace supra
{
	class ConfigurationEntryType
	{
	public:
		virtual ~ConfigurationEntryType() {}
	};

	template <typename ValueType>
	class ConfigurationEntry : public ConfigurationEntryType
	{
	public:
		ConfigurationEntry(ValueType value) : m_valueEntry(value) {}
		virtual ~ConfigurationEntry() {}

		virtual const ValueType& get() const { return m_valueEntry; }

	private:
		ValueType m_valueEntry;
	};

	class ConfigurationDictionary
	{
	public:
		ConfigurationDictionary()
			:p_valueRangeDictionary(nullptr) {};
		ConfigurationDictionary(const tinyxml2::XMLElement* parentXmlElement);

		const ConfigurationDictionary& operator=(const ConfigurationDictionary& a)
		{
			m_mapEntries = a.m_mapEntries;
			return *this;
		}

		void setValueRangeDictionary(const ValueRangeDictionary* valueRangeDictionary)
		{
			p_valueRangeDictionary = valueRangeDictionary;
		}

		template <typename ValueType>
		void set(const std::string& key, const ValueType& value) {
			auto valueEntry = std::make_shared<ConfigurationEntry<ValueType> >(value);
			m_mapEntries[key] = valueEntry;
		}


		template <typename ValueType>
		ValueType get(const std::string& key, const ValueType& defaultValue) const {
			auto iteratorValue = m_mapEntries.find(key);
			if (iteratorValue != m_mapEntries.end())
			{
				auto pValueEntry = iteratorValue->second;
				const ConfigurationEntry<ValueType> * pValueTyped = dynamic_cast<const ConfigurationEntry<ValueType> *>(pValueEntry.get());
				if (pValueTyped)
				{
					return pValueTyped->get();
				}
				else if (p_valueRangeDictionary)
				{
					return p_valueRangeDictionary->getDefaultValue<ValueType>(key);
				}
				else
				{
					return defaultValue;
				}
			}
			else if (p_valueRangeDictionary && p_valueRangeDictionary->hasKey(key))
			{
				return p_valueRangeDictionary->getDefaultValue<ValueType>(key);
			}
			else
			{
				return defaultValue;
			}
		}

		template <typename ValueType>
		ValueType get(const std::string& key) const {
			auto iteratorValue = m_mapEntries.find(key);
			if (iteratorValue != m_mapEntries.end())
			{
				auto pValueEntry = iteratorValue->second;
				const ConfigurationEntry<ValueType> * pValueTyped = dynamic_cast<const ConfigurationEntry<ValueType> *>(pValueEntry.get());
				if (pValueTyped)
				{
					return pValueTyped->get();
				}
				else if (p_valueRangeDictionary)
				{
					return p_valueRangeDictionary->getDefaultValue<ValueType>(key);
				}
				else
				{
					logging::log_error("Trying to access parameter '", key, "' without value and valueRangeDictionary");
					return TemplateTypeDefault<ValueType>::getDefault();
				}
			}
			else if (p_valueRangeDictionary)
			{
				return p_valueRangeDictionary->getDefaultValue<ValueType>(key);
			}
			else
			{
				logging::log_error("Trying to access parameter '", key, "' without value and valueRangeDictionary");
				return TemplateTypeDefault<ValueType>::getDefault();
			}
		}

		void checkEntriesAndLog(const std::string & nodeID)
		{
			if (p_valueRangeDictionary)
			{
				std::vector<std::string> toRemove;
				for (auto entry : m_mapEntries)
				{
					bool valueGood = false;
					if (p_valueRangeDictionary->hasKey(entry.first))
					{
						switch (p_valueRangeDictionary->getType(entry.first))
						{
						case TypeBool:
							valueGood = checkEntryAndLogTemplated<bool>(entry.first, nodeID);
							break;
						case TypeInt8:
							valueGood = checkEntryAndLogTemplated<int8_t>(entry.first, nodeID);
							break;
						case TypeUint8:
							valueGood = checkEntryAndLogTemplated<uint8_t>(entry.first, nodeID);
							break;
						case TypeInt16:
							valueGood = checkEntryAndLogTemplated<int16_t>(entry.first, nodeID);
							break;
						case TypeUint16:
							valueGood = checkEntryAndLogTemplated<uint16_t>(entry.first, nodeID);
							break;
						case TypeInt32:
							valueGood = checkEntryAndLogTemplated<int32_t>(entry.first, nodeID);
							break;
						case TypeUint32:
							valueGood = checkEntryAndLogTemplated<uint32_t>(entry.first, nodeID);
							break;
						case TypeInt64:
							valueGood = checkEntryAndLogTemplated<int64_t>(entry.first, nodeID);
							break;
						case TypeUint64:
							valueGood = checkEntryAndLogTemplated<uint64_t>(entry.first, nodeID);
							break;
						case TypeFloat:
							valueGood = checkEntryAndLogTemplated<float>(entry.first, nodeID);
							break;
						case TypeDouble:
							valueGood = checkEntryAndLogTemplated<double>(entry.first, nodeID);
							break;
						case TypeString:
							valueGood = checkEntryAndLogTemplated<std::string>(entry.first, nodeID);
							break;
						case TypeDataType:
							valueGood = checkEntryAndLogTemplated<DataType>(entry.first, nodeID);
							break;
						case TypeUnknown:
						default:
							logging::log_error("cannot check validity of configuration entry '", entry.first, "' as its range type is unknown.");
						}
					}
					if (!valueGood)
					{
						toRemove.push_back(entry.first);
					}
				}

				//remove all determined bad values
				for (std::string keyToRemove : toRemove)
				{
					m_mapEntries.erase(keyToRemove);
					logging::log_warn("Removed configuration entry '", keyToRemove, "' because it's type or value were not as defined by the range.");
				}
			}
			else
			{
				logging::log_error("Configuration checking failed, due to missing valueRangeDictionary");
				//we cannot verify any entry. Remove all!
				m_mapEntries.clear();
			}
		}

		template <typename ValueType>
		bool checkEntryAndLogTemplated(const std::string& key, const std::string & nodeID)
		{
			auto iteratorValue = m_mapEntries.find(key);
			if (iteratorValue != m_mapEntries.end())
			{
				auto pValueEntry = iteratorValue->second;
				const ConfigurationEntry<ValueType> * pValueTyped = dynamic_cast<const ConfigurationEntry<ValueType> *>(pValueEntry.get());
				if (pValueTyped)
				{
					bool entryGood = p_valueRangeDictionary->isInRange<ValueType>(key, pValueTyped->get());
					if (entryGood)
					{
						logging::log_parameter("Parameter: ", nodeID, ".", key, " = ", pValueTyped->get());
					}
					else {
						logging::log_log("Rejected out-of-range parameter: ", nodeID, ".", key, " = ", pValueTyped->get());
					}
					return entryGood;
				}
				else
				{
					//Types do not fit
					logging::log_log("Rejected parameter of wrong type for: ", nodeID, ".", key);
					return false;
				}
			}
			else {
				//The key is not even present
				logging::log_log("Rejected unknown parameter: ", nodeID, ".", key);
				return false;
			}
		}

	private:
		std::map<std::string, std::shared_ptr<ConfigurationEntryType> > m_mapEntries;

		const ValueRangeDictionary* p_valueRangeDictionary;
	};
}

#endif //!__CONFIGURATIONDICTIONARY_H__
