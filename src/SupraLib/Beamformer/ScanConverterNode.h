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
	private:
		typedef tbb::flow::broadcast_node<std::shared_ptr<RecordObject> > MaskOutputNodeType;

	public:
		ScanConverterNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 2; }

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
			if (index == 1)
			{
				return m_maskOutputNode.get();
			}
			return nullptr;
		};
	protected:
		void configurationEntryChanged(const std::string& configKey);
		void configurationChanged();

	private:
		template <typename T>
		std::shared_ptr<RecordObject> convertTemplated(const std::shared_ptr<USImage> pInImage);
		std::shared_ptr<RecordObject> checkTypeAndConvert(const std::shared_ptr<RecordObject> mainObj);
		void sendMask(const std::shared_ptr<RecordObject> pImage);

		std::unique_ptr<tbb::flow::graph_node> m_node;
		std::unique_ptr<MaskOutputNodeType> m_maskOutputNode;

		std::mutex m_mutex;
		bool m_maskSent;

		bool m_parameterChangeRequiresInternalUpdate;
		bool m_forceImageResolution;
		double m_imageResolution;
		DataType m_outputType;

		std::unique_ptr<ScanConverter> m_converter;

		std::shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		std::shared_ptr<USImageProperties> m_scanConvImageProperties;
	};
}

#endif //!__SCANCONVERTERNODE_H__
