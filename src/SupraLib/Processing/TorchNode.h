// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __TORCHNODE_H__
#define __TORCHNODE_H__

#ifdef HAVE_TORCH

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

#include <AbstractNode.h>
#include <RecordObject.h>
#include <Container.h>
#include <vec.h>

namespace supra
{
	class USImageProperties;
	class TorchInference;
}

// To include the node fully, add it in src/SupraLib/CMakeLists.txt and "InterfaceFactory::createNode"!

namespace supra
{
	class TorchNode : public AbstractNode {
	public:
		TorchNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

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
		std::shared_ptr<RecordObject> checkTypeAndProcess(std::shared_ptr<RecordObject> mainObj);
		template <typename InputType>
		std::shared_ptr<ContainerBase> processTemplateSelection(
			std::shared_ptr<const Container<InputType> > imageData,
			vec3s inputSize,
			vec3s outputSize,
			const std::string& currentLayout,
			const std::string& finalLayout);
		void loadModule();

		std::unique_ptr<tbb::flow::graph_node> m_node;
		std::mutex m_mutex;
		std::shared_ptr<TorchInference> m_torchModule;

		std::string m_modelFilename;
		DataType m_modelInputDataType;
		DataType m_modelOutputDataType;
		std::string m_modelInputClass;
		std::string m_modelOutputClass;
		std::string m_modelInputLayout;
		std::string m_modelOutputLayout;
		DataType m_nodeOutputDataType;
		uint32_t m_inferencePatchSize;
		uint32_t m_inferencePatchOverlap;
	};
}

#endif //HAVE_TORCH

#endif //!__TORCHNODE_H__
