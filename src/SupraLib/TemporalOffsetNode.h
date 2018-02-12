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

#ifndef __TEMPORALOFFSETNODE_H__
#define __TEMPORALOFFSETNODE_H__

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
	class TemporalOffsetNode : public AbstractNode {
	public:
		typedef tbb::flow::function_node<std::shared_ptr<RecordObject>, std::shared_ptr<RecordObject>> nodeType;

	public:
		TemporalOffsetNode(tbb::flow::graph& graph, const std::string & nodeID);

		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 1; }

		virtual tbb::flow::receiver<std::shared_ptr<RecordObject> > * getInput(size_t index) {
			if (index == 0)
			{
				return &m_node;
			}
			return nullptr;
		};

		virtual tbb::flow::sender<std::shared_ptr<RecordObject> > * getOutput(size_t index) {
			if (index == 0)
			{
				return &m_node;
			}
			return nullptr;
		};
	private:
		std::shared_ptr<RecordObject> addOffset(std::shared_ptr<RecordObject> mainObj);

		nodeType m_node;
		
		double m_offset;
		
		std::mutex m_objectMutex;

	protected:
		//Needs to be thread safe
		virtual void configurationEntryChanged(const std::string& configKey);
		//Needs to be thread safe
		virtual void configurationChanged();
	};
}

#endif //!__TEMPORALOFFSETNODE_H__
