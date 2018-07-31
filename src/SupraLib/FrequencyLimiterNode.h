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

#ifndef __FREQUENCYLIMITERNODE_H__
#define __FREQUENCYLIMITERNODE_H__

#include <memory>
#include <vector>
#include <mutex>
#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "RecordObject.h"

namespace supra
{
	class FrequencyLimiterNode : public AbstractNode {
	private:
		typedef tbb::flow::broadcast_node<std::shared_ptr<RecordObject> > outputNodeType;

	public:
		FrequencyLimiterNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 1; }

		virtual tbb::flow::graph_node * getInput(size_t index) {
			if (index == 0)
			{
				return m_inputNode.get();
			}
			return nullptr;
		};

		virtual tbb::flow::graph_node * getOutput(size_t index) {
			if (index == 0)
			{
				return m_outputNode.get();
			}
			return nullptr;
		};
	private:
		void forwardMessage(std::shared_ptr<RecordObject> obj);

		std::unique_ptr<tbb::flow::graph_node> m_inputNode;
		std::unique_ptr<outputNodeType> m_outputNode;

		double m_maxFrequency;
		double m_lastMessageTimestamp;

		std::mutex m_objectMutex;

	protected:
		//Needs to be thread safe
		virtual void configurationEntryChanged(const std::string& configKey);
		//Needs to be thread safe
		virtual void configurationChanged();
	};
}

#endif //!__FREQUENCYLIMITERNODE_H__
