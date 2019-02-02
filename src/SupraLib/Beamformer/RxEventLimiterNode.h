// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2018, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __RXEVENTLIMITERNODE_H__
#define __RXEVENTLIMITERNODE_H__

#ifdef HAVE_CUDA

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

#include <AbstractNode.h>
#include <RecordObject.h>
#include <Container.h>
#include <vec.h>

// forward declaraions
namespace supra
{
	class USImageProperties;
	class RxBeamformerParameters;
	class USRawData;
}

namespace supra
{
	class RxEventLimiterNode : public AbstractNode {
	public:
		RxEventLimiterNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

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
		std::shared_ptr<ContainerBase> processTemplateSelection(std::shared_ptr<const USRawData > imageData);
		void updateImagePropertiesAndRxBeamformerParameters(
			std::shared_ptr<const USImageProperties> newImageProperties, 
			std::shared_ptr<const RxBeamformerParameters> newRxBeamformerParameters);

		std::unique_ptr<tbb::flow::graph_node> m_node;

		std::mutex m_mutex;

		std::shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		std::shared_ptr<USImageProperties> m_modifiedImageProperties;
		std::shared_ptr<const RxBeamformerParameters> m_lastSeenRxBeamformerParameters;
		std::shared_ptr<RxBeamformerParameters> m_modifiedRxBeamformerParameters;
		bool m_parametersRequireUpdate;

		uint32_t m_firstEventIdxToKeep;
		uint32_t m_lastEventIdxToKeep;
	};
}

#endif //HAVE_CUDA

#endif //!__RXEVENTLIMITERNODE_H__
