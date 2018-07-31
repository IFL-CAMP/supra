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

#include "FrequencyLimiterNode.h"

#include <utilities/Logging.h>
#include <utilities/CallFrequency.h>
using namespace std;

namespace supra
{
	FrequencyLimiterNode::FrequencyLimiterNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_lastMessageTimestamp(0)
	{
		if (queueing)
		{
			m_inputNode = unique_ptr<NodeTypeOneSidedQueueing>(
				new NodeTypeOneSidedQueueing(graph, 1, [this](shared_ptr<RecordObject> obj) { forwardMessage(obj); }));
		}
		else
		{
			m_inputNode = unique_ptr<NodeTypeOneSidedDiscarding>(
				new NodeTypeOneSidedDiscarding(graph, 1, [this](shared_ptr<RecordObject> obj) { forwardMessage(obj); }));
		}
		m_outputNode = unique_ptr<outputNodeType>(new outputNodeType(graph));

		m_callFrequency.setName("Limiter");

		m_valueRangeDictionary.set<double>("maxFrequency", 0.001, 100, 10, "Maximum Frequency [Hz]");
		configurationChanged();
	}

	void FrequencyLimiterNode::forwardMessage(shared_ptr<RecordObject> obj)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);

		//prepare the list of synced Objects
		double curTimestamp = obj->getSyncTimestamp();

		if (curTimestamp - m_lastMessageTimestamp > 1 / m_maxFrequency)
		{
			m_outputNode->try_put(obj);
			m_lastMessageTimestamp = curTimestamp;
		}
	}

	void FrequencyLimiterNode::configurationEntryChanged(const std::string & configKey)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);
		if (configKey == "maxFrequency")
		{
			m_maxFrequency = m_configurationDictionary.get<double>("maxFrequency");
		}
	}

	void FrequencyLimiterNode::configurationChanged()
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);

		m_maxFrequency = m_configurationDictionary.get<double>("maxFrequency");
	}
}