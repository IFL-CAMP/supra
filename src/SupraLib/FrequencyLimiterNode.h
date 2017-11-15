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
	public:
		typedef tbb::flow::function_node<std::shared_ptr<RecordObject>, tbb::flow::continue_msg, TBB_QUEUE_RESOLVER(false)> inputNodeType;
		typedef tbb::flow::broadcast_node<std::shared_ptr<RecordObject> > outputNodeType;

	public:
		FrequencyLimiterNode(tbb::flow::graph& graph, const std::string & nodeID);

		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 1; }

		virtual tbb::flow::receiver<std::shared_ptr<RecordObject> > * getInput(size_t index) {
			if (index == 0)
			{
				return &m_inputNode;
			}
			return nullptr;
		};

		virtual tbb::flow::sender<std::shared_ptr<RecordObject> > * getOutput(size_t index) {
			if (index == 0)
			{
				return &m_outputNode;
			}
			return nullptr;
		};
	private:
		void forwardMessage(std::shared_ptr<RecordObject> obj);

		inputNodeType m_inputNode;
		outputNodeType m_outputNode;

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
