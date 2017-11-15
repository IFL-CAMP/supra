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

#include "AutoQuitNode.h"

#include "SupraManager.h"
#include <utilities/Logging.h>
#include <utilities/CallFrequency.h>
using namespace std;

namespace supra
{
	AutoQuitNode::AutoQuitNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_inputNode(graph, 1, [this](shared_ptr<RecordObject> obj) { countMessage(obj); })
		, m_messagesReceived(0)
	{
		m_callFrequency.setName("AutoQuit");

		m_valueRangeDictionary.set<uint32_t>("maxMessage", 0, 1000000, 1000, "Maximum number of messages");
		configurationChanged();
	}

	void AutoQuitNode::countMessage(shared_ptr<RecordObject> obj)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);

		m_messagesReceived++;

		if (m_messagesReceived == m_maxMessageNum)
		{
			//stop inputs
			SupraManager::Get()->quit();
		}
	}

	void AutoQuitNode::configurationEntryChanged(const std::string & configKey)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);
		if (configKey == "maxMessage")
		{
			m_maxMessageNum = m_configurationDictionary.get<uint32_t>("maxMessage");
		}
	}

	void AutoQuitNode::configurationChanged()
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);

		m_maxMessageNum = m_configurationDictionary.get<uint32_t>("maxMessage");
	}
}