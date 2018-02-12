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

#include "TemporalOffsetNode.h"

#include <utilities/Logging.h>
#include <utilities/CallFrequency.h>
#include <algorithm>
using namespace std;

namespace supra
{
	TemporalOffsetNode::TemporalOffsetNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_node(graph, 1, [this](shared_ptr<RecordObject> mainObj) -> shared_ptr<RecordObject> { return addOffset(mainObj); })
	{
		m_callFrequency.setName("TempOffset");

		m_valueRangeDictionary.set<double>("offset", -10.0, 10.0, 0.0, "Temporal offset [s]");
		configurationChanged();
	}

	shared_ptr<RecordObject> TemporalOffsetNode::addOffset(shared_ptr<RecordObject> obj)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);

		obj->setSyncOffset(m_offset);
		return obj;
	}

	void TemporalOffsetNode::configurationEntryChanged(const std::string & configKey)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);
		if (configKey == "offset")
		{
			m_offset = m_configurationDictionary.get<double>("offset");
		}
	}

	void TemporalOffsetNode::configurationChanged()
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);
		m_offset = m_configurationDictionary.get<double>("offset");
	}
}