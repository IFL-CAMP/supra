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

#ifndef __VALUERANGEDICTIONARY_H__
#define __VALUERANGEDICTIONARY_H__

#include <string>
#include <memory>
#include <tuple>
#include <map>
#include <vector>
#include <algorithm>
#include <utilities/TemplateTypeDefault.h>
#include <utilities/DataType.h>

namespace supra
{
	/// Base class for the templated \see ValueRangeEntry
	class ValueRangeType
	{
	public:
		virtual ~ValueRangeType() {}
		/// Returns the type of the parameter described by the value range
		virtual DataType getType() const {
			return TypeUnknown;
		};
	};

	/// Describes a node parameter with its valid range. Is part of the parameter system.
	/// There are three types of ranges: 
	/// -Discrete: Only values that are in a fixed vector are valid
	/// -Closed Range: Values between the defined upper and lower bound are valid
	/// -Unrestricted: All values are valid
	/// This is selected on construction by the choice of constructor
	template <typename ValueType>
	class ValueRangeEntry : public ValueRangeType
	{
	public:
		/// Constructor for a discrete range, takes a vector of the allowed values.
		/// Additionally takes the parameter's default value and its display name
		ValueRangeEntry(const std::vector<ValueType>& discreteRange, const ValueType& defaultValue, const std::string& displayName)
			: m_discreteRange(discreteRange)
			, m_continuousRange()
			, m_isContinuous(false)
			, m_isUnrestricted(false)
			, m_defaultValue(defaultValue)
			, m_displayName(displayName) {}
		/// Constructor for a closed range, takes the lower and upper bound of the allowed values.
		/// Additionally takes the parameter's default value and its display name
		ValueRangeEntry(const ValueType& lowerBound, const ValueType& upperBound, const ValueType& defaultValue, const std::string& displayName)
			: m_discreteRange()
			, m_continuousRange({ lowerBound, upperBound })
			, m_isContinuous(true)
			, m_isUnrestricted(false)
			, m_defaultValue(defaultValue)
			, m_displayName(displayName) {}
		/// Constructor for an unrestricted range,
		/// takes the parameter's default value and its display name
		ValueRangeEntry(const ValueType& defaultValue, const std::string& displayName)
			: m_discreteRange()
			, m_continuousRange()
			, m_isContinuous(false)
			, m_isUnrestricted(true)
			, m_defaultValue(defaultValue)
			, m_displayName(displayName) {}
		virtual ~ValueRangeEntry() {}

		/// Return the type of the parameter range
		virtual DataType getType() const { return DataTypeGet<ValueType>(); }
		/// Returns whether the range is unrestricted
		virtual bool isUnrestricted() const { return m_isUnrestricted; }
		/// Returns whether the range is closed but continous
		virtual bool isContinuous() const { return m_isContinuous; }
		/// Returns the allowed values of this range. Should only be called for discrete ranges.
		virtual const std::vector<ValueType>& getDiscrete() const { return m_discreteRange; }
		/// Returns the upper and lower bound of this range. Should only be called for closed ranges.
		virtual const std::pair<ValueType, ValueType>& getContinuous() const { return m_continuousRange; }
		/// Returns this parameter's default value
		virtual const ValueType & getDefaultValue() const { return m_defaultValue; }
		/// Checks whether the value is within the range, that is whether it is valid
		virtual bool isInRange(const ValueType& valueToCheck) const {
			bool valIsInRange = false;
			if (m_isUnrestricted)
			{
				valIsInRange = true;
			}
			else if (m_isContinuous)
			{
				valIsInRange =
					valueToCheck >= m_continuousRange.first &&
					valueToCheck <= m_continuousRange.second;
			}
			else {
				valIsInRange =
					std::find(
						m_discreteRange.begin(),
						m_discreteRange.end(),
						valueToCheck)
					!= m_discreteRange.end();
			}
			return valIsInRange;
		}
		/// Returns the display name of the parameter
		const std::string& getDisplayName() const { return m_displayName; }

	private:
		bool m_isUnrestricted;
		bool m_isContinuous;
		std::vector<ValueType> m_discreteRange;
		std::pair<ValueType, ValueType> m_continuousRange;
		ValueType m_defaultValue;
		std::string m_displayName;
	};

	/// A collection of parameter ranges \see ValueRangeEntry, every node has one
	class ValueRangeDictionary
	{
	public:
		/// Default constructor
		ValueRangeDictionary() {};

		/// Assignment operator. Copies all ranges defined in the assignee
		const ValueRangeDictionary& operator=(const ValueRangeDictionary& a)
		{
			m_mapEntries = a.m_mapEntries;
			return *this;
		}

		/// Creates a discrete range for the given parameter, takes a vector of the allowed values.
		/// Additionally takes the parameter's default value and its display name
		template <typename ValueType>
		void set(const std::string& key, const std::vector<ValueType>& value, const ValueType& defaultValue, const std::string& displayName) {
			auto valueEntry = std::make_shared<ValueRangeEntry<ValueType> >(value, defaultValue, displayName);
			m_mapEntries[key] = valueEntry;
		}

		/// Creates a closed range for the given parameter, takes the lower and upper bound of the allowed values.
		/// Additionally takes the parameter's default value and its display name
		template <typename ValueType>
		void set(const std::string& key, const ValueType& lowerBound, const ValueType& upperBound, const ValueType& defaultValue, const std::string& displayName) {
			auto valueEntry = std::make_shared<ValueRangeEntry<ValueType> >(lowerBound, upperBound, defaultValue, displayName);
			m_mapEntries[key] = valueEntry;
		}

		/// Creates an unrestricted rangefor the given parameter,
		/// takes the parameter's default value and its display name
		template <typename ValueType>
		void set(const std::string& key, const ValueType& defaultValue, const std::string& displayName) {
			auto valueEntry = std::make_shared<ValueRangeEntry<ValueType> >(defaultValue, displayName);
			m_mapEntries[key] = valueEntry;
		}

		/// Checks whether a range for a parameter is defined in this dictionary
		bool hasKey(const std::string& key) const
		{
			return m_mapEntries.find(key) != m_mapEntries.end();
		}

		/// Returns a list of the parameter ids that are defined in this dictionary
		std::vector<std::string> getKeys() const
		{
			std::vector<std::string> keys(m_mapEntries.size());
			std::transform(m_mapEntries.begin(), m_mapEntries.end(), keys.begin(),
				[](std::pair<std::string, std::shared_ptr<ValueRangeType> > mapPair) -> std::string {return mapPair.first; });

			return keys;
		}

		/// Returns the type of a parameter. Only defined if `hasKey(key) == true`
		DataType getType(std::string key) const
		{
			auto iteratorValue = m_mapEntries.find(key);
			return iteratorValue->second->getType();
		}

		/// Returns the range for a parameter
		template <typename ValueType>
		const ValueRangeEntry<ValueType> * get(const std::string& key) const {
			auto iteratorValue = m_mapEntries.find(key);
			auto pValueEntry = iteratorValue->second;
			const ValueRangeEntry<ValueType> * pValueTyped = dynamic_cast<const ValueRangeEntry<ValueType> *>(pValueEntry.get());
			return pValueTyped;
		}

		void remove(const std::string& key) {
			m_mapEntries.erase(key);
		}

		/// Returns the default value of a parameter.
		/// If the parameter has no range in this dictionary, the types default values is returned
		/// ( \see TemplateTypeDefault)
		template <typename ValueType>
		ValueType getDefaultValue(const std::string& key) const {
			auto iteratorValue = m_mapEntries.find(key);
			if (iteratorValue == m_mapEntries.end())
			{
				return TemplateTypeDefault<ValueType>::getDefault();
			}
			auto pValueEntry = iteratorValue->second;
			const ValueRangeEntry<ValueType> * pValueTyped = dynamic_cast<const ValueRangeEntry<ValueType> *>(pValueEntry.get());
			if (pValueTyped)
			{
				return pValueTyped->getDefaultValue();
			}
			else {
				return TemplateTypeDefault<ValueType>::getDefault();
			}
		}

		/// Checks whether the value is within the range, that is whether it is valid
		/// Returns false, if the parameter is not defined in this dictionary
		template <typename ValueType>
		bool isInRange(const std::string& key, const ValueType& value) const
		{
			bool result = false;
			auto iteratorValue = m_mapEntries.find(key);
			if (iteratorValue != m_mapEntries.end())
			{
				auto pValueEntry = iteratorValue->second;
				const ValueRangeEntry<ValueType> * pValueTyped = dynamic_cast<const ValueRangeEntry<ValueType> *>(pValueEntry.get());
				result = pValueTyped->isInRange(value);
			}
			return result;
		}

	private:
		std::map<std::string, std::shared_ptr<ValueRangeType> > m_mapEntries;
	};
}

#endif //!__VALUERANGEDICTIONARY_H__
