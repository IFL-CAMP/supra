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

#ifndef __SUPRAMANAGER_H__
#define __SUPRAMANAGER_H__

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <thread>

#include "RecordObject.h"
#include "AbstractInput.h"
#include "AbstractOutput.h"

namespace supra
{
	/*! \brief The main interface to the supra-lib.
	*         Creates and keeps track of all nodes in the processing graph and can be used to create connections between them.
	*/
	class SupraManager
	{
	public:
		static std::shared_ptr<SupraManager> Get();

		//Config file handling
		void readFromXml(const char* configXmlFilename, bool queueing = false);
		void readInputDevicesFromXml(tinyxml2::XMLElement* inputsElement);
		void readOutputDevicesFromXml(tinyxml2::XMLElement* outputsElement, bool queueing);
		void readNodesFromXml(tinyxml2::XMLElement* nodesElement, bool queueing);
		void readConnectionsFromXml(tinyxml2::XMLElement* connectionsElement);
		void writeToXml(std::string configXmlFilename);
		void writeInputDevicesToXml(tinyxml2::XMLNode* devicesElement);
		void writeOutputDevicesToXml(tinyxml2::XMLNode* devicesElement);
		void writeNodesToXml(tinyxml2::XMLNode* devicesElement);
		void writeConnectionsToXml(tinyxml2::XMLNode* rootElement);

		//Node access
		std::vector<std::string> getInputDeviceIDs();
		std::vector<std::string> getOutputDeviceIDs();
		std::vector<std::string> getNodeIDs();
		std::map<std::string, std::string> getNodeTypes();
		std::shared_ptr<AbstractNode> getNode(std::string nodeID);
		std::shared_ptr<AbstractInput> getInputDevice(std::string nodeID);
		std::shared_ptr<AbstractOutput> getOutputDevice(std::string nodeID);
		bool nodeExists(std::string nodeID);
		std::vector<std::tuple<std::string, size_t, std::string, size_t> > getNodeConnections();
		std::map<std::string, std::vector<size_t> > getImageOutputPorts();
		std::map<std::string, std::vector<size_t> > getTrackingOutputPorts();

		//Node modifications
		bool addNode(std::string nodeID, std::shared_ptr<AbstractNode> node, std::string nodeType);
		std::string addNode(std::string nodeType, bool queueing = false);

		template <class nodeToConstruct, typename... constructorArgTypes>
		bool addNodeConstruct(std::string nodeID, std::string nodeType, constructorArgTypes... constructorArgs)
		{
			std::shared_ptr<AbstractNode> newNode = std::shared_ptr<AbstractNode>(
				new nodeToConstruct(*m_graph, nodeID, constructorArgs...));

			return addNode(nodeID, newNode, nodeType);
		}

		bool removeNode(std::string nodeID);

		void connect(std::string fromID, size_t fromPort, std::string toID, size_t toPort);
		void disconnect(std::string fromID, size_t fromPort, std::string toID, size_t toPort);
		void startOutputs();
		void startOutputsSequence();
		void stopOutputsSequence();
		void startInputs();
		void stopAndWaitInputs();
		void waitInputs();

		void freezeInputs();
		void unfreezeInputs();
		bool inputsFrozen();

		int32_t getFreezeTimeout();
		void setFreezeTimeout(int32_t timeout);
		int32_t resetFreezeTimeout();
		
		//wait for complete graph to be finished
		void waitForGraph();

		// request the application to be stopped
		void quit();

		void setQuitCallback(std::function<void(void)> quitCallback);

	private:
		SupraManager();
		void freezeThread();
		std::string findUnusedID(std::string prefix);

		std::shared_ptr<tbb::flow::graph> m_graph;
		std::map<std::string, std::shared_ptr<AbstractInput> > m_inputDevices;
		std::map<std::string, std::shared_ptr<AbstractOutput> > m_outputDevices;
		std::map<std::string, std::shared_ptr<AbstractNode> > m_nodes;
		std::map<std::string, std::string > m_nodeTypes;
		std::map<std::tuple<std::string, size_t, std::string, size_t>, bool> m_nodeConnections;
		std::vector<std::shared_ptr<AbstractNode> > m_removedNodes;

		std::unique_ptr<std::thread> m_freezeThread;
		std::atomic<int32_t> m_freezeTimeout;
		std::atomic<int32_t> m_freezeTimeoutInit;
		std::atomic_bool m_inputsFrozen;
		std::atomic_bool m_freezeThreadContinue;

		std::function<void(void)> m_quitCallback;
	};
}

#endif //!__SUPRAMANAGER_H__
