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
		typedef tbb::flow::function_node<std::shared_ptr<RecordObject>, std::shared_ptr<RecordObject>> mainNodeType;
		typedef tbb::flow::function_node<std::shared_ptr<RecordObject>, tbb::flow::continue_msg> toSyncNodeType;

	public:
		StreamSynchronizer(tbb::flow::graph& graph, const std::string & nodeID);

		virtual size_t getNumInputs() { return m_toSyncNodes.size() + 1; }
		virtual size_t getNumOutputs() { return 1; }

		virtual tbb::flow::receiver<std::shared_ptr<RecordObject> > * getInput(size_t index) {
			if (index == 0)
			{
				return &m_mainNode;
			}
			else if (index <= m_toSyncNodes.size())
			{
				return &m_toSyncNodes[index - 1];
			}
			return nullptr;
		};

		virtual tbb::flow::sender<std::shared_ptr<RecordObject> > * getOutput(size_t index) {
			if (index == 0)
			{
				return &m_mainNode;
			}
			return nullptr;
		};
	private:
		static bool syncObjectComparator(const std::shared_ptr<RecordObject> & a, const std::shared_ptr<RecordObject> & b);

		std::shared_ptr<SyncRecordObject> findSynced(std::shared_ptr<RecordObject> mainObj);
		void addToSyncList(size_t channel, std::shared_ptr<RecordObject> syncObj);

		mainNodeType m_mainNode;
		std::vector<toSyncNodeType> m_toSyncNodes;

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
