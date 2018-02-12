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

#include "StreamSynchronizer.h"

#include <utilities/Logging.h>
#include <utilities/CallFrequency.h>
#include <algorithm>
using namespace std;

namespace supra
{
	StreamSynchronizer::StreamSynchronizer(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_mainNode(graph, 1, [this](shared_ptr<RecordObject> mainObj) -> shared_ptr<SyncRecordObject> { return findSynced(mainObj); })
		, m_graphHandle(graph)
	{
		m_callFrequency.setName("Sync");

		m_valueRangeDictionary.set<uint32_t>("numStreamsToSync", 1, "Num synced streams");
		m_valueRangeDictionary.set<double>("maxTimeToKeep", 1 * 1e6, 20 * 1e6, 7 * 1e6, "Temporal window");
		configurationChanged();
	}

	shared_ptr<SyncRecordObject> StreamSynchronizer::findSynced(shared_ptr<RecordObject> mainObj)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);

		//prepare the list of synced Objects
		vector<shared_ptr<const RecordObject> > synced;

		m_callFrequency.measure();
		for (size_t i = 0; i < m_syncLists.size(); i++)
		{
			auto pSyncList = &m_syncLists[i];
			lock_guard<mutex> syncListLock(*m_syncListMutexes[i].get());
			if (pSyncList->size() > 0)
			{
				auto firstElementOlder = upper_bound(pSyncList->begin(), pSyncList->end(),
					mainObj, &StreamSynchronizer::syncObjectComparator);
				if (firstElementOlder == pSyncList->end())
				{
					synced.push_back(*(pSyncList->end() - 1));
				}
				else if (firstElementOlder == pSyncList->begin())
				{
					synced.push_back(*(pSyncList->begin()));
				}
				else {
					double timeToFoundElement = ((*firstElementOlder)->getSyncTimestamp() - mainObj->getSyncTimestamp());
					double timeToElementBefore = (mainObj->getSyncTimestamp() - (*(firstElementOlder - 1))->getSyncTimestamp());
					if (timeToFoundElement < timeToElementBefore)
					{
						synced.push_back(*firstElementOlder);
					}
					else {
						synced.push_back(*(firstElementOlder - 1));
					}
				}
			}
		}
		m_callFrequency.measureEnd();
		return make_shared<SyncRecordObject>(mainObj, synced, mainObj->getReceiveTimestamp(), mainObj->getSyncTimestamp());
	}

	void StreamSynchronizer::addToSyncList(size_t channel, shared_ptr<RecordObject> syncObj)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);

		static CallFrequency sl("SyncList");
		lock_guard<mutex> syncListLock(*m_syncListMutexes[channel].get());
		if (m_syncLists[channel].size() > 0)
		{
			//get the latest currently stored timestamp
			double latestTimestamp = (*(m_syncLists[channel].end() - 1))->getSyncTimestamp();
			if (latestTimestamp >= syncObj->getSyncTimestamp())
			{
				logging::log_log("StreamSynchronizer (channel ", channel, ") did not accept sync object with timestamp ", syncObj->getSyncTimestamp(),
					"because it is older than the latest contained timestamp ", latestTimestamp);
				return;
			}
		}

		m_syncLists[channel].push_back(syncObj);

		//remove old objects
		shared_ptr<RecordObject> timeHolder = make_shared<RecordObject>(syncObj->getReceiveTimestamp() - m_maxTimeToKeep, syncObj->getSyncTimestamp() - m_maxTimeToKeep);
		auto firstElementToKeep = lower_bound(m_syncLists[channel].begin(), m_syncLists[channel].end(),
			timeHolder, &StreamSynchronizer::syncObjectComparator);
		if (firstElementToKeep != m_syncLists[channel].begin())
		{
			m_syncLists[channel].erase(m_syncLists[channel].begin(), firstElementToKeep);
		}
		sl.measure();
	}

	void StreamSynchronizer::configurationEntryChanged(const std::string & configKey)
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);
	}

	void StreamSynchronizer::configurationChanged()
	{
		std::lock_guard<std::mutex> objectLock(m_objectMutex);

		m_numStreamsToSync = m_configurationDictionary.get<uint32_t>("numStreamsToSync");
		m_maxTimeToKeep = m_configurationDictionary.get<double>("maxTimeToKeep");

		//setup the sync nodes
		m_syncLists.resize(m_numStreamsToSync);
		m_toSyncNodes.clear();
		for (size_t i = 0; i < m_numStreamsToSync; i++)
		{
			m_syncListMutexes.push_back(unique_ptr<mutex>(new mutex()));
			m_toSyncNodes.emplace_back(m_graphHandle, 1,
				[this, i](shared_ptr<RecordObject> objToSync) { addToSyncList(i, objToSync); });
		}
	}

	bool StreamSynchronizer::syncObjectComparator(const shared_ptr<RecordObject>& a, const shared_ptr<RecordObject>& b)
	{
		return a->getSyncTimestamp() < b->getSyncTimestamp();
	}
}