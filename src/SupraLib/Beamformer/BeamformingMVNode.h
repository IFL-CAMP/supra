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

#ifndef __BEAMFORMINGMVNODE_H__
#define __BEAMFORMINGMVNODE_H__

#ifdef HAVE_BEAMFORMER_MINIMUM_VARIANCE

#include <memory>
//#include <vector>
//#include <deque>
//#include <mutex>
#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "RecordObject.h"

#include <cublas_v2.h>

//#include "SyncRecordObject.h"
//#include "RxBeamformerParameters.h"

namespace supra
{
	//forward declarations
	//enum WindowType : uint32_t;
	class USImageProperties;

	class BeamformingMVNode : public AbstractNode {
	public:
		typedef tbb::flow::function_node<std::shared_ptr<RecordObject>, std::shared_ptr<RecordObject>, TBB_QUEUE_RESOLVER(false)> nodeType;

	public:
		BeamformingMVNode(tbb::flow::graph& graph, const std::string & nodeID);
		~BeamformingMVNode();

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
		void configurationChanged();
		void configurationEntryChanged(const std::string& configKey);

	private:
		std::shared_ptr<RecordObject> checkTypeAndBeamform(std::shared_ptr<RecordObject> mainObj);
		void updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties);

		std::shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		std::shared_ptr<USImageProperties> m_editedImageProperties;

		std::mutex m_mutex;
		cublasHandle_t m_cublasH;

		nodeType m_node;

		uint32_t m_subArraySize;
		uint32_t m_temporalSmoothing;
	};
}

#endif //HAVE_BEAMFORMER_MINIMUM_VARIANCE

#endif //!__BEAMFORMINGMVNODE_H__
