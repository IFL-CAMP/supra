// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2017, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __STREAMSYNCNODE_H__
#define __STREAMSYNCNODE_H__

#include "AbstractNode.h"
#include "RecordObject.h"

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

namespace supra
{
	class StreamSyncNode : public AbstractNode {
	public:
		StreamSyncNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 1; }

		virtual tbb::flow::graph_node * getInput(size_t index) {
			if (index == 0)
			{
				return m_node.get();
			}
			return nullptr;
		};

		virtual tbb::flow::graph_node * getOutput(size_t index) {
			if (index == 0)
			{
				return m_node.get();
			}
			return nullptr;
		};
	protected:
		void configurationEntryChanged(const std::string& configKey);
		void configurationChanged();

	private:
		std::shared_ptr<RecordObject> checkTypeAndSynchronize(std::shared_ptr<RecordObject> mainObj);

		std::unique_ptr<tbb::flow::graph_node> m_node;
	};
}

#endif //!__STREAMSYNCNODE_H__
