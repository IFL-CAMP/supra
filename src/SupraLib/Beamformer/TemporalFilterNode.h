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

#ifndef __TEMPORALFILTERNODE_H__
#define __TEMPORALFILTERNODE_H__

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "RecordObject.h"
#include "TemporalFilter.h"

#include <queue>

namespace supra
{
	class TemporalFilterNode : public AbstractNode {
	public:
		typedef tbb::flow::function_node<std::shared_ptr<RecordObject>, std::shared_ptr<RecordObject>, TBB_QUEUE_RESOLVER(false)> nodeType;

	public:
		TemporalFilterNode(tbb::flow::graph& graph, const std::string & nodeID);

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
	protected:
		void configurationEntryChanged(const std::string& configKey);
		void configurationChanged();

	private:
		std::shared_ptr<RecordObject> filter(std::shared_ptr<RecordObject> mainObj);
		void updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties);

		nodeType m_node;

		std::mutex m_mutex;

		std::unique_ptr<TemporalFilter> m_temporalFilter;
		std::queue<std::shared_ptr<const Container<int16_t> > > m_storedImages;
		uint32_t m_numImages;
		vec3s m_imageSize;

		std::shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		std::shared_ptr<const USImageProperties> m_editedImageProperties;
	};
}

#endif //!__TEMPORALFILTERNODE_H__
