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

#ifndef __ABSTRACTNODE_H__
#define __ABSTRACTNODE_H__

#include <memory>
#include <tbb/flow_graph.h>

#include "ConfigurationDictionary.h"
#include "ValueRangeDictionary.h"

#include "utilities/CallFrequency.h"

//This is ugly, but "queueing" and "rejecting" are enum-values in old versions and types in newer versions...
#if TBB_INTERFACE_VERSION_MAJOR >= 9
#define TBB_QUEUE_RESOLVER(__buffering__) typename std::conditional<__buffering__, tbb::flow::queueing, tbb::flow::rejecting>::type
#else
#define TBB_QUEUE_RESOLVER(__buffering__) (__buffering__ ? tbb::flow::queueing : tbb::flow::rejecting)
#endif //TBB_INTERFACE_VERSION_MAJOR >= 9

namespace supra
{
	class RecordObject;
}

namespace supra
{
	/*! \brief Abstract interface for a general node (input, output or processing).
	*
	*  This is the common interface for all nodes.
	*/
	class AbstractNode
	{
	public:
		/// Base constructor for all nodes
		AbstractNode(const std::string & nodeID)
			:m_nodeID(nodeID)
		{
			m_configurationDictionary.setValueRangeDictionary(&m_valueRangeDictionary);
		}

		virtual ~AbstractNode() {}

		/// Returns the number of input ports this node maintains
		/// This method is overwritten in each implementation
		virtual size_t getNumInputs() = 0;
		/// Returns the number of output ports this node maintains
		/// This method is overwritten in each implementation
		virtual size_t getNumOutputs() = 0;

		/// Returns a pointer to the input port with the given number
		virtual tbb::flow::receiver<std::shared_ptr<RecordObject> > *
			getInput(size_t index) {
			return nullptr;
		}

		/// Returns a pointer to the output port with the given number
		virtual tbb::flow::sender<std::shared_ptr<RecordObject> > *
			getOutput(size_t index) {
			return nullptr;
		}

		/// Returns a const pointer to the \see ValueRangeDictionary of this node
		/// The ValueRangeDictionary describes the parameters of the node and 
		/// their valid ranges
		const ValueRangeDictionary * getValueRangeDictionary()
		{
			return &m_valueRangeDictionary;
		}

		/// Returns a const pointer to the \see ConfigurationDictionary of this node
		/// The ConfigurationDictionary contains the parameters currently set 
		/// and their values
		const ConfigurationDictionary * getConfigurationDictionary()
		{
			return &m_configurationDictionary;
		}

		/// Returns the ID of the node. 
		/// Node IDs are unique and cannot be changed after creation.
		const std::string& getNodeID()
		{
			return m_nodeID;
		}

		/// Templated interface to change a node parameter
		/// returns whether the value was valid
		template<typename ValueType>
		bool changeConfig(const std::string& configKey, const ValueType& newValue)
		{
			if (m_valueRangeDictionary.hasKey(configKey) &&
				m_valueRangeDictionary.isInRange(configKey, newValue))
			{
				logging::log_parameter("Parameter: ", m_nodeID, ".", configKey, " = ", newValue);
				m_configurationDictionary.set(configKey, newValue);
				configurationEntryChanged(configKey);
				return true;
			}
			return false;
		}
		/// Function to set the whole \see ConfiguraitonDictionary at once
		/// Only parameters whose value is valid are applied
		void changeConfig(const ConfigurationDictionary& newConfig)
		{
			// trigger callback for major configuration changes in overloaded implementation
			configurationDictionaryChanged(newConfig);

			//validate the configuration entries
			ConfigurationDictionary validConfig = newConfig;
			validConfig.setValueRangeDictionary(&m_valueRangeDictionary);
			validConfig.checkEntriesAndLog(m_nodeID);

			//store all valid entries
			m_configurationDictionary = validConfig;
			configurationChanged();
		}

		/// Returns a string with the timing info (call frequency and run-time)
		/// if the node uses the \see CallFrequency member to monitor itself
		std::string getTimingInfo()
		{
			return m_callFrequency.getTimingInfo();
		}

	protected:
		/// The collection of node parameters
		ConfigurationDictionary m_configurationDictionary;
		/// The definition of parameters and their respective ranges
		ValueRangeDictionary m_valueRangeDictionary;
		/// \see CallFrequency can be used by the node implementation to monitor its
		/// timings (frequency of activation and run-time)
		CallFrequency m_callFrequency;

	protected:
		/// Callback for the node implementation to be notified of the change of parameters.
		/// Needs to be overwritten and thread-safe
		virtual void configurationEntryChanged(const std::string& configKey) {};
		/// Callback for the node implementation to be notified of the change of parameters.
		/// Needs to be overwritten and thread-safe
		virtual void configurationChanged() {};
		/// Callback for the node implementation to be notified of the change of a full dictionary change.
		/// can be be overwritten but must be thread-safe
		/// dictionary does not contain fail-safe mechanisms yet, thus should be only used to adjust value range changes or add new settings on the fly
		virtual void configurationDictionaryChanged(const ConfigurationDictionary& newConfig) {};


	private:
		std::string m_nodeID;
	};
}

#endif //!__ABSTRACTNODE_H__
