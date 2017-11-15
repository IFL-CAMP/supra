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

#ifndef __INTERFACEFACTORY_H__
#define __INTERFACEFACTORY_H__

#include <memory>

#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "AbstractInput.h"
#include "AbstractOutput.h"

namespace supra
{
	class InterfaceFactory {
	public:
		static std::shared_ptr<tbb::flow::graph> createGraph();
		static std::shared_ptr<AbstractInput<RecordObject> > createInputDevice(std::shared_ptr<tbb::flow::graph> pG, const std::string& nodeID, std::string deviceType);
		static std::shared_ptr<AbstractOutput> createOutputDevice(std::shared_ptr<tbb::flow::graph> pG, const std::string & nodeID, std::string deviceType);
		static std::shared_ptr<AbstractNode> createNode(std::shared_ptr<tbb::flow::graph> pG, const std::string & nodeID, std::string nodeType);
	};
}

#endif //!__INTERFACEFACTORY_H__
