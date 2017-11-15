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

#ifndef __SCANCONVERTERNODE_H__
#define __SCANCONVERTERNODE_H__

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "RecordObject.h"
#include "ScanConverter.h"

namespace supra
{
	class ScanConverterNode : public AbstractNode {
	public:
		typedef tbb::flow::function_node<std::shared_ptr<RecordObject>, std::shared_ptr<RecordObject>, TBB_QUEUE_RESOLVER(false)> NodeType;
		typedef tbb::flow::broadcast_node<std::shared_ptr<RecordObject> > MaskOutputNodeType;

	public:
		ScanConverterNode(tbb::flow::graph& graph, const std::string & nodeID);

		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 2; }

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
			if (index == 1)
			{
				return &m_maskOutputNode;
			}
			return nullptr;
		};
	protected:
		void configurationEntryChanged(const std::string& configKey);
		void configurationChanged();

	private:
		template <typename T>
		std::shared_ptr<RecordObject> convertTemplated(std::shared_ptr<USImage<T> > pInImage);
		std::shared_ptr<RecordObject> checkTypeAndConvert(std::shared_ptr<RecordObject> mainObj);
		void sendMask(std::shared_ptr<RecordObject> pImage);

		NodeType m_node;
		MaskOutputNodeType m_maskOutputNode;

		std::mutex m_mutex;
		bool m_maskSent;

		bool m_parameterChangeRequiresInternalUpdate;
		bool m_forceImageResolution;
		double m_imageResolution;

		std::unique_ptr<ScanConverter> m_converter;

		std::shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		std::shared_ptr<USImageProperties> m_scanConvImageProperties;
	};
}

#endif //!__SCANCONVERTERNODE_H__
