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

#ifndef __RAWDELAYNODE_H__
#define __RAWDELAYNODE_H__

#ifdef HAVE_BEAMFORMER

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "RecordObject.h"
#include "SyncRecordObject.h"
#include "RxBeamformerParameters.h"
#include "RawDelay.h"

namespace supra
{
	//forward declarations
	enum WindowType : uint32_t;
	class USImageProperties;

	class RawDelayNode : public AbstractNode {
	public:
		RawDelayNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

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
		std::shared_ptr<RecordObject> checkTypeAndDelay(std::shared_ptr<RecordObject> mainObj);
		template <typename RawElementType>
		std::shared_ptr<USRawData> delay(std::shared_ptr<const USRawData> mainObj);
		void readWindowType();
		void updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties);

		std::shared_ptr<RawDelay> m_rawDelayCuda;
		std::shared_ptr<const RxBeamformerParameters> m_lastSeenBeamformerParameters;

		std::shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		std::shared_ptr<USImageProperties> m_editedImageProperties;

		std::mutex m_mutex;

		std::unique_ptr<tbb::flow::graph_node> m_node;
		double m_fNumber;
		WindowType m_windowType;
		DataType m_outputType;
		double m_windowParameter;
		double m_speedOfSoundMMperS;
	};
}

#endif //HAVE_BEAMFORMER
#endif //!__RAWDELAYNODE_H__
