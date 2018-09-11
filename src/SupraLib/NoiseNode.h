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

#ifndef __NOISENODE_H__
#define __NOISENODE_H__

#ifdef HAVE_CUDA

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

#include <AbstractNode.h>
#include <RecordObject.h>
#include <Container.h>
#include <vec.h>

// To include the node fully, add it in src/SupraLib/CMakeLists.txt and "InterfaceFactory::createNode"!

namespace supra
{
	class NoiseNode : public AbstractNode {
	public:
		NoiseNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

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
		std::shared_ptr<ContainerBase> processTemplateSelection(std::shared_ptr<const Container<InputType> > imageData, vec3s size);

		std::unique_ptr<tbb::flow::graph_node> m_node;

		std::mutex m_mutex;

		double m_additiveUniformMin;
		double m_additiveUniformMax;
		double m_additiveGaussMean;
		double m_additiveGaussStd;
		double m_multiplicativeUniformMin;
		double m_multiplicativeUniformMax;
		double m_multiplicativeGaussMean;
		double m_multiplicativeGaussStd;
		bool m_additiveUniformCorrelated;
		bool m_additiveGaussCorrelated;
		bool m_multiplicativeUniformCorrelated;
		bool m_multiplicativeGaussCorrelated;

		DataType m_outputType;
	};
}

#endif //HAVE_CUDA

#endif //!__NOISENODE_H__
