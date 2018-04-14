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

#ifndef __STREAMSYNCHRONIZER_H__
#define __STREAMSYNCHRONIZER_H__

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "RecordObject.h"
#include "SyncRecordObject.h"

namespace supra
{
	class StreamSynchronizer : public AbstractNode {
	public:
		StreamSynchronizer(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

		virtual size_t getNumInputs() { return m_toSyncNodes.size() + 1; }
		virtual size_t getNumOutputs() { return 1; }

		virtual tbb::flow::graph_node * getInput(size_t index) {
			if (index == 0)
			{
				return m_mainNode.get();
			}
			else if (index <= m_toSyncNodes.size())
			{
				return m_toSyncNodes[index - 1].get();
			}
			return nullptr;
		};

		virtual tbb::flow::graph_node * getOutput(size_t index) {
			if (index == 0)
			{
				return m_mainNode.get();
			}
			return nullptr;
		};
	private:
		static bool syncObjectComparator(const std::shared_ptr<RecordObject> & a, const std::shared_ptr<RecordObject> & b);

		std::shared_ptr<SyncRecordObject> findSynced(std::shared_ptr<RecordObject> mainObj);
		void addToSyncList(size_t channel, std::shared_ptr<RecordObject> syncObj);

		std::unique_ptr<tbb::flow::graph_node> m_mainNode;
		std::vector<std::unique_ptr<tbb::flow::graph_node> > m_toSyncNodes;

		std::vector<std::deque<std::shared_ptr<RecordObject>>> m_syncLists;
		std::vector<std::unique_ptr<std::mutex> > m_syncListMutexes;

		double m_maxTimeToKeep;
		uint32_t m_numStreamsToSync;

		std::mutex m_objectMutex;
		tbb::flow::graph & m_graphHandle;

	protected:
		//Needs to be thread safe
		virtual void configurationEntryChanged(const std::string& configKey);
		//Needs to be thread safe
		virtual void configurationChanged();
	};
}

#endif //!__STREAMSYNCHRONIZER_H__
