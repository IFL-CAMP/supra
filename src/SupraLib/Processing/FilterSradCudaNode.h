// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Walter Simson 
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __FILTERSRADCUDANODE_H__
#define __FILTERSRADCUDANODE_H__

#ifdef HAVE_CUDA

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

#include <AbstractNode.h>
#include <RecordObject.h>
#include <Container.h>
#include <vec.h>

// To include the node fully, add it in src/SupraLib/CMakeLists.txt and "InterfaceFactory::createNode"!

// This node implements a speckle filter following
// Yongjian Yu and S. T. Acton, "Speckle reducing anisotropic diffusion,"
// in IEEE Transactions on Image Processing, vol. 11, no. 11, pp. 1260-1270, Nov. 2002.
// https://doi.org/10.1109/TIP.2002.804276
namespace supra
{
	class FilterSradCudaNode : public AbstractNode {
	public:
		FilterSradCudaNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

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

		double m_eps = M_EPS;
		uint32_t m_numberIterations;
		double m_lambda; // step size of PDE solver
		double m_speckleScale;  // speckle q_0 analog to equation 37 in "the paper" 
		double m_speckleScaleDecay;  // Decay of speckle scale over time (rho, e1. 37)
		DataType m_outputType;
	};
}

#endif //HAVE_CUDA

#endif //!__FILTERSRADCUDANODE_H__
