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
		TemporalFilterNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

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
		std::shared_ptr<RecordObject> filter(std::shared_ptr<RecordObject> mainObj);
		template <typename InputType>
		std::shared_ptr<ContainerBase> filterTemplated(
			const std::queue<std::shared_ptr<const ContainerBase> > & inImageData,
			vec3s size,
			const std::vector<double> weights);
		void updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties);

		std::unique_ptr<tbb::flow::graph_node> m_node;

		std::mutex m_mutex;

		std::unique_ptr<TemporalFilter> m_temporalFilter;
		std::queue<std::shared_ptr<const ContainerBase> > m_storedImages;
		uint32_t m_numImages;
		DataType m_outputType;
		vec3s m_imageSize;
		DataType m_imageDataType;

		std::shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		std::shared_ptr<const USImageProperties> m_editedImageProperties;
	};
}

#endif //!__TEMPORALFILTERNODE_H__
