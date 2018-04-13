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

#ifndef __BEAMFORMINGNODE_H__
#define __BEAMFORMINGNODE_H__

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "RecordObject.h"
#include "SyncRecordObject.h"
#include "RxBeamformerParameters.h"
#include "RxBeamformerCuda.h"

namespace supra
{
	//forward declarations
	enum WindowType : uint32_t;
	class USImageProperties;

	class BeamformingNode : public AbstractNode {
	public:
		BeamformingNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

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
		void configurationChanged();
		void configurationEntryChanged(const std::string& configKey);

	private:
		std::shared_ptr<RecordObject> checkTypeAndBeamform(std::shared_ptr<RecordObject> mainObj);
		template <typename InputType>
		std::shared_ptr<USImage> beamformTemplated(std::shared_ptr<const USRawData> pRawData);
		void readWindowType();
		void readBeamformerType();
		void updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties);

		std::shared_ptr<RxBeamformerCuda> m_beamformer;
		std::shared_ptr<const RxBeamformerParameters> m_lastSeenBeamformerParameters;

		std::shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		std::shared_ptr<USImageProperties> m_editedImageProperties;

		std::mutex m_mutex;

		std::unique_ptr<tbb::flow::graph_node> m_node;
		double m_fNumber;
		WindowType m_windowType;
		double m_windowParameter;
		DataType m_outputType;
		RxBeamformerCuda::RxSampleBeamformer m_beamformerType;
		bool m_interpolateTransmits;
	};
}

#endif //!__BEAMFORMINGNODE_H__
