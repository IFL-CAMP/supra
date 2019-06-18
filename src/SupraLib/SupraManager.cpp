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

#include "SupraManager.h"

#include "utilities/tinyxml2/tinyxml2.h"
#include "utilities/Logging.h"
#include "InterfaceFactory.h"

#include <string>
#include <chrono>

using namespace tinyxml2;
using namespace std;

namespace supra
{
	SupraManager::SupraManager()
	{
		m_graph = InterfaceFactory::createGraph();
		m_freezeTimeoutInit = 5 * 60; // seconds
		m_inputsFrozen = false;
		resetFreezeTimeout();
	}

	std::shared_ptr<SupraManager> SupraManager::Get()
	{
		static std::shared_ptr<SupraManager> instance = std::shared_ptr<SupraManager>(new SupraManager());
		return instance;
	}

	void SupraManager::readFromXml(const char * configXmlFilename, bool queueing)
	{
		tinyxml2::XMLDocument doc;
		XMLError errLoad = doc.LoadFile(configXmlFilename);
		logging::log_error_if(errLoad != tinyxml2::XML_SUCCESS,
			"SupraManager::readFromXml: Could not open file '", configXmlFilename, "'. Error value was: ", errLoad);

		//Read the list of objects from the XML file
		XMLElement* configElement = doc.FirstChildElement("supra_config");
		logging::log_error_if(!configElement,
			"SupraManager::readFromXml: Error reading config file '", configXmlFilename, "'. It did not contain the root element <supra_config>");
		if (configElement)
		{
			XMLElement* devicesElement = configElement->FirstChildElement("devices");
			//Inputs
			if (devicesElement)
			{
				XMLElement* inputsElement = devicesElement->FirstChildElement("inputs");

				if (inputsElement)
				{
					readInputDevicesFromXml(inputsElement);
				}
				//Outputs
				XMLElement* outputsElement = devicesElement->FirstChildElement("outputs");
				if (outputsElement)
				{
					readOutputDevicesFromXml(outputsElement, queueing);
				}
				//Other nodes
				XMLElement* nodesElement = devicesElement->FirstChildElement("nodes");
				if (nodesElement)
				{
					readNodesFromXml(nodesElement, queueing);
				}
			}

			//Read the edges from the XML file
			XMLElement* connectionsElement = configElement->FirstChildElement("connections");
			if (connectionsElement)
			{
				readConnectionsFromXml(connectionsElement);
			}
		}
	}

	void SupraManager::readInputDevicesFromXml(tinyxml2::XMLElement * inputsElement)
	{
		XMLElement* nextInput = inputsElement->FirstChildElement("input");
		while (nextInput)
		{
			//find type and id of input element
			string inputType = nextInput->Attribute("type");
			string inputID = nextInput->Attribute("id");

			size_t numPorts = 1;
			if (nextInput->Attribute("ports"))
			{
				numPorts = std::stoi(nextInput->Attribute("ports"));
			}

			//create input element
			auto in = InterfaceFactory::createInputDevice(m_graph, inputID, inputType, numPorts);

			if (in)
			{
				//load config for this element
				ConfigurationDictionary dict(nextInput);
				in->changeConfig(dict);

				//store input node
				bool couldAdd = addNode(inputID, in, inputType);
				logging::log_warn_if(!couldAdd, "SupraManager: Node '", inputID, "' already existed. Did not add it to collection.");
				if (couldAdd)
				{
					m_inputDevices[inputID] = in;
				}
			}

			//advance to next element
			nextInput = nextInput->NextSiblingElement("input");
		}
	}

	void SupraManager::readOutputDevicesFromXml(tinyxml2::XMLElement * outputsElement, bool queueing)
	{
		XMLElement* nextOutput = outputsElement->FirstChildElement("output");
		while (nextOutput)
		{
			//find type and id of input element
			string outputType = nextOutput->Attribute("type");
			string outputID = nextOutput->Attribute("id");

			//create input element
			auto out = InterfaceFactory::createOutputDevice(m_graph, outputID, outputType, queueing);

			if (out)
			{
				//load config for this element
				ConfigurationDictionary dict(nextOutput);
				out->changeConfig(dict);

				//store output node
				bool couldAdd = addNode(outputID, out, outputType);
				logging::log_warn_if(!couldAdd, "SupraManager: Node '", outputID, "' already existed. Did not add it to collection.");
				if (couldAdd)
				{
					m_outputDevices[outputID] = out;
				}
			}

			//advance to next element
			nextOutput = nextOutput->NextSiblingElement("output");
		}
	}

	void SupraManager::readNodesFromXml(tinyxml2::XMLElement * nodesElement, bool queueing)
	{
		XMLElement* nextNode = nodesElement->FirstChildElement("node");
		while (nextNode)
		{
			//find type and id of node element
			string nodeType = nextNode->Attribute("type");
			string nodeID = nextNode->Attribute("id");

			//create node
			auto node = InterfaceFactory::createNode(m_graph, nodeID, nodeType, queueing);

			if (node)
			{
				//load parameters for this element
				ConfigurationDictionary dict(nextNode);
				node->changeConfig(dict);

				//store node
				bool couldAdd = addNode(nodeID, node, nodeType);
				logging::log_warn_if(!couldAdd, "SupraManager: Node '", nodeID, "' already existed. Did not add it to collection.");
			}

			//advance to next element
			nextNode = nextNode->NextSiblingElement("node");
		}
	}

	void SupraManager::readConnectionsFromXml(tinyxml2::XMLElement * connectionsElement)
	{
		XMLElement* nextConnection = connectionsElement->FirstChildElement("connection");
		while (nextConnection)
		{
			//from
			XMLElement* fromElement = nextConnection->FirstChildElement("from");
			string fromID = fromElement->Attribute("id");
			int fromPort = 0;
			if (fromElement->QueryIntAttribute("port", &fromPort) != XML_SUCCESS)
			{
				logging::log_error("SupraManager: Error parsing the port attribute of a connection from '", fromID, "'.");
			}

			//to
			XMLElement* toElement = nextConnection->FirstChildElement("to");
			string toID = toElement->Attribute("id");
			int toPort = 0;
			if (toElement->QueryIntAttribute("port", &toPort) != XML_SUCCESS)
			{
				logging::log_error("SupraManager: Error parsing the port attribute of a connection to '", toID, "'.");
			}

			//create the connection
			connect(fromID, fromPort, toID, toPort);

			//advance to next element
			nextConnection = nextConnection->NextSiblingElement("connection");
		}
	}

	void SupraManager::writeToXml(std::string configXmlFilename) 
	{
		tinyxml2::XMLDocument doc;
		
		doc.InsertFirstChild(doc.NewDeclaration());
		auto rootElement = doc.InsertEndChild(doc.NewElement("supra_config"));
		auto devicesElement = rootElement->InsertEndChild(doc.NewElement("devices"));

		writeInputDevicesToXml(devicesElement);
		writeOutputDevicesToXml(devicesElement);
		writeNodesToXml(devicesElement);
		writeConnectionsToXml(rootElement);

		doc.SaveFile(configXmlFilename.c_str());
	}

	void SupraManager::writeInputDevicesToXml(tinyxml2::XMLNode* devicesElement)
	{
		auto doc = devicesElement->GetDocument();
		auto inputsElement = devicesElement->InsertEndChild(doc->NewElement("inputs"));

		for (auto inputDevicePair : m_inputDevices)
		{
			string nodeID = inputDevicePair.first;
			auto inputDevice = inputDevicePair.second;

			auto inputElement = doc->NewElement("input");
			inputsElement->InsertEndChild(inputElement);
			inputElement->SetAttribute("type", m_nodeTypes[nodeID].c_str());
			inputElement->SetAttribute("id", nodeID.c_str());
			if (inputDevice->getNumOutputs() > 1)
			{
				inputElement->SetAttribute("ports", static_cast<unsigned int>(inputDevice->getNumOutputs()));
			}
			inputDevice->getConfigurationDictionary()->toXml(inputElement);
		}
	}

	void SupraManager::writeOutputDevicesToXml(tinyxml2::XMLNode* devicesElement)
	{		
		auto doc = devicesElement->GetDocument();
		auto outputsElement = devicesElement->InsertEndChild(doc->NewElement("outputs"));

		for (auto outputDevicePair : m_outputDevices)
		{
			string nodeID = outputDevicePair.first;
			auto outputDevice = outputDevicePair.second;

			auto outputElement = doc->NewElement("output");
			outputsElement->InsertEndChild(outputElement);
			outputElement->SetAttribute("type", m_nodeTypes[nodeID].c_str());
			outputElement->SetAttribute("id", nodeID.c_str());
			outputDevice->getConfigurationDictionary()->toXml(outputElement);
		}
	}

	void SupraManager::writeNodesToXml(tinyxml2::XMLNode* devicesElement)
	{
		auto doc = devicesElement->GetDocument();
		auto nodesElement = devicesElement->InsertEndChild(doc->NewElement("nodes"));

		for (auto nodePair : m_nodes)
		{
			string nodeID = nodePair.first;
			auto node = nodePair.second;

			if (m_inputDevices.find(nodeID) == m_inputDevices.end() && m_outputDevices.find(nodeID) == m_outputDevices.end())
			{
				auto nodeElement = doc->NewElement("node");
				nodesElement->InsertEndChild(nodeElement);
				nodeElement->SetAttribute("type", m_nodeTypes[nodeID].c_str());
				nodeElement->SetAttribute("id", nodeID.c_str());
				node->getConfigurationDictionary()->toXml(nodeElement);
			}
		}
	}

	void SupraManager::writeConnectionsToXml(tinyxml2::XMLNode* rootElement)
	{
		auto doc = rootElement->GetDocument();
		auto connectionsElement = rootElement->InsertEndChild(doc->NewElement("connections"));

		auto connections = SupraManager::getNodeConnections();
		for (auto connection : connections)
		{
			auto connectionElement = connectionsElement->InsertEndChild(doc->NewElement("connection"));
			auto fromElement = doc->NewElement("from");
			auto toElement = doc->NewElement("to");
			connectionElement->InsertEndChild(fromElement);
			connectionElement->InsertEndChild(toElement);

			fromElement->SetAttribute("id", get<0>(connection).c_str());
			fromElement->SetAttribute("port", static_cast<unsigned int>(get<1>(connection)));
			toElement->SetAttribute("id", get<2>(connection).c_str());
			toElement->SetAttribute("port", static_cast<unsigned int>(get<3>(connection)));
		}
	}

	std::vector<std::string> SupraManager::getInputDeviceIDs()
	{
		std::vector<std::string> nodeIDs(m_inputDevices.size());
		std::transform(m_inputDevices.begin(), m_inputDevices.end(), nodeIDs.begin(),
			[](pair<string, shared_ptr<AbstractInput> > mapPair) -> std::string {return mapPair.first; });
		return nodeIDs;
	}

	std::vector<std::string> SupraManager::getOutputDeviceIDs()
	{
		std::vector<std::string> nodeIDs(m_outputDevices.size());
		std::transform(m_outputDevices.begin(), m_outputDevices.end(), nodeIDs.begin(),
			[](pair<string, shared_ptr<AbstractOutput> > mapPair) -> std::string {return mapPair.first; });
		return nodeIDs;
	}

	std::vector<std::string> SupraManager::getNodeIDs()
	{
		std::vector<std::string> nodeIDs(m_nodes.size());
		std::transform(m_nodes.begin(), m_nodes.end(), nodeIDs.begin(),
			[](pair<string, shared_ptr<AbstractNode> > mapPair) -> std::string {return mapPair.first; });
		return nodeIDs;
	}

	std::map<std::string, std::string> SupraManager::getNodeTypes()
	{
		return m_nodeTypes;
	}

	shared_ptr<AbstractNode> SupraManager::getNode(string nodeID)
	{
		shared_ptr<AbstractNode> retVal = shared_ptr<AbstractNode>(nullptr);
		if (nodeExists(nodeID))
		{
			retVal = m_nodes[nodeID];
		}
		return retVal;
	}

	shared_ptr<AbstractInput> SupraManager::getInputDevice(string nodeID)
	{
		shared_ptr<AbstractInput> retVal = shared_ptr<AbstractInput>(nullptr);
		if (m_inputDevices.find(nodeID) != m_inputDevices.end())
		{
			retVal = m_inputDevices[nodeID];
		}
		return retVal;
	}

	shared_ptr<AbstractOutput> SupraManager::getOutputDevice(string nodeID)
	{
		shared_ptr<AbstractOutput> retVal = shared_ptr<AbstractOutput>(nullptr);
		if (m_outputDevices.find(nodeID) != m_outputDevices.end())
		{
			retVal = m_outputDevices[nodeID];
		}
		return retVal;
	}

	bool SupraManager::addNode(string nodeID, shared_ptr<AbstractNode> node, string nodeType)
	{
		if (nodeExists(nodeID))
		{
			return false;
		}
		else
		{
			m_nodes[nodeID] = node;
			m_nodeTypes[nodeID] = nodeType;
			logging::log_log("SupraManager: Added Node '", nodeID, "'.");
			return true;
		}
	}

	std::string SupraManager::addNode(std::string nodeType, bool queueing)
	{
		string newID = findUnusedID(nodeType);

		//create node
		auto node = InterfaceFactory::createNode(m_graph, newID, nodeType, queueing);

		bool couldAdd = false;
		if (node)
		{
			//store node
			couldAdd = addNode(newID, node, nodeType);
			logging::log_warn_if(!couldAdd, "SupraManager: Node '", newID, "' already existed. Did not add it to collection.");
		}
		if (!couldAdd)
		{
			newID = "";
		}
		return newID;
	}

	bool SupraManager::nodeExists(string nodeID)
	{
		return (m_nodes.find(nodeID) != m_nodes.end());
	}

	std::vector<std::tuple<std::string, size_t, std::string, size_t> > SupraManager::getNodeConnections()
	{
		std::vector<std::tuple<std::string, size_t, std::string, size_t> > nodeConnections(m_nodeConnections.size());
		std::transform(m_nodeConnections.begin(), m_nodeConnections.end(), nodeConnections.begin(),
			[](const std::pair<std::tuple<std::string, size_t, std::string, size_t>, bool>& mapPair) -> 
				std::tuple<std::string, size_t, std::string, size_t> {return mapPair.first; });
		return nodeConnections;
	}

	std::map<std::string, std::vector<size_t>> SupraManager::getImageOutputPorts()
	{
		std::map<std::string, std::vector<size_t>> map;
		for (auto inputDevicePair : m_inputDevices)
		{
			string nodeID = inputDevicePair.first;
			auto portList = inputDevicePair.second->getImageOutputPorts();
			if (portList.size() > 0)
			{
				map[nodeID] = portList;
			}
		}
		return map;
	}

	std::map<std::string, std::vector<size_t>> SupraManager::getTrackingOutputPorts()
	{
		std::map<std::string, std::vector<size_t>> map;
		for (auto inputDevicePair : m_inputDevices)
		{
			string nodeID = inputDevicePair.first;
			auto portList = inputDevicePair.second->getTrackingOutputPorts();
			if (portList.size() > 0)
			{
				map[nodeID] = portList;
			}
		}
		return map;
	}

	bool SupraManager::removeNode(string nodeID)
	{
		if (!nodeExists(nodeID))
		{
			return false;
		}
		else
		{
			m_removedNodes.push_back(m_nodes[nodeID]);
			m_nodes.erase(nodeID);
			return true;
		}
	}

	void SupraManager::connect(string fromID, size_t fromPort, string toID, size_t toPort)
	{
		if (nodeExists(fromID) && nodeExists(toID))
		{
			shared_ptr<AbstractNode> fromNode = m_nodes[fromID];
			shared_ptr<AbstractNode> toNode = m_nodes[toID];
			if (fromNode->getNumOutputs() > fromPort && toNode->getNumInputs() > toPort)
			{
				auto connTuple = std::make_tuple(fromID, fromPort, toID, toPort);
				if (m_nodeConnections.count(connTuple) == 0)
				{
					m_nodeConnections[connTuple] = true;
					tbb::flow::make_edge(
						*(dynamic_cast<tbb::flow::sender<std::shared_ptr<RecordObject> >*>(fromNode->getOutput(fromPort))),
						*(dynamic_cast<tbb::flow::receiver<std::shared_ptr<RecordObject> >*>(toNode->getInput(toPort))));
					logging::log_log("SupraManager: Added connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, ").");
				}
				else
				{
					logging::log_error("SupraManager: Could not add connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). It already exists.");
				}
			}
			else {
				logging::log_error("SupraManager: Could not add connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). One of the ports does not exist.");
			}
		}
		else {
			logging::log_error("SupraManager: Could not add connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). One node does not exist.");
		}
	}

	void SupraManager::disconnect(string fromID, size_t fromPort, string toID, size_t toPort)
	{
		if (nodeExists(fromID) && nodeExists(toID))
		{
			shared_ptr<AbstractNode> fromNode = m_nodes[fromID];
			shared_ptr<AbstractNode> toNode = m_nodes[toID];
			if (fromNode->getNumOutputs() > fromPort && toNode->getNumInputs() > toPort)
			{
				auto connTuple = std::make_tuple(fromID, fromPort, toID, toPort);
				if (m_nodeConnections.count(connTuple) == 1 &&
					m_nodeConnections[connTuple])
				{
					m_nodeConnections.erase(connTuple);
					tbb::flow::remove_edge(
						*(dynamic_cast<tbb::flow::sender<std::shared_ptr<RecordObject> >*>(fromNode->getOutput(fromPort))),
						*(dynamic_cast<tbb::flow::receiver<std::shared_ptr<RecordObject> >*>(toNode->getInput(toPort))));
					logging::log_log("SupraManager: Removed connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, ").");
				}
				else
				{
					logging::log_error("SupraManager: Could not remove connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). It does not exist.");
				}
			}
			else {
				logging::log_error("SupraManager: Could not remove connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). One of the ports does not exist.");
			}
		}
		else {
			logging::log_error("SupraManager: Could not remove connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). One node does not exist.");
		}
	}

	void SupraManager::startOutputs()
	{
		for (auto outputDevicePair : m_outputDevices)
		{
			string nodeID = outputDevicePair.first;
			logging::log_log("SupraManager: Starting output '", nodeID, "'");
			outputDevicePair.second->initializeOutput();
			if (outputDevicePair.second->ready())
			{
				outputDevicePair.second->setRunning(true);
			}
			else {
				logging::log_warn("SupraManager: Output '", nodeID, "' not started. It was not ready.");
			}
		}
	}

	void SupraManager::startOutputsSequence()
	{
		for (auto outputDevicePair : m_outputDevices)
		{
			string nodeID = outputDevicePair.first;
			logging::log_log("SupraManager: Starting sequence at '", nodeID, "'");
			outputDevicePair.second->startSequence();
		}
	}

	void SupraManager::stopOutputsSequence()
	{
		for (auto outputDevicePair : m_outputDevices)
		{
			string nodeID = outputDevicePair.first;
			logging::log_log("SupraManager: Stopping sequence at '", nodeID, "'");
			outputDevicePair.second->stopSequence();
		}
	}

	void SupraManager::startInputs()
	{
		// Start the timeout thread that will freeze the inputs
		m_freezeThreadContinue = true;
		m_freezeThread = std::unique_ptr<std::thread>(new std::thread([this] {this->freezeThread(); }));

		for (auto inputDevicePair : m_inputDevices)
		{
			string nodeID = inputDevicePair.first;
			logging::log_log("SupraManager: Starting input '", nodeID, "'");
			inputDevicePair.second->initializeDevice();
			if (inputDevicePair.second->ready())
			{
				inputDevicePair.second->start();
			}
			else {
				logging::log_warn("SupraManager: Input '", nodeID, "' not started. It was not ready.");
			}
		}
	}

	void SupraManager::stopAndWaitInputs()
	{
		for (auto inputDevicePair : m_inputDevices)
		{
			if (inputDevicePair.second->getRunning())
			{
				string nodeID = inputDevicePair.first;
				logging::log_log("SupraManager: Stopping input '", nodeID, "'");
				inputDevicePair.second->setRunning(false);
			}
		}
		waitInputs();
	}

	void SupraManager::waitInputs()
	{
		for (auto inputDevicePair : m_inputDevices)
		{
			if (inputDevicePair.second->getRunning())
			{
				string nodeID = inputDevicePair.first;
				logging::log_log("SupraManager: Waiting for input '", nodeID, "' to finish.");
				inputDevicePair.second->waitForFinish();
			}
		}

		m_freezeThreadContinue = false;
		if (m_freezeThread)
		{
			if (m_freezeThread->joinable())
			{
				m_freezeThread->join();
			}
			m_freezeThread = nullptr;
		}
	}

	void supra::SupraManager::freezeInputs()
	{
		// If inputs have not been frozen already
		bool expected = false;
		if (m_inputsFrozen.compare_exchange_strong(expected, true))
		{
			for (auto inputDevicePair : m_inputDevices)
			{
				if (inputDevicePair.second->getRunning())
				{
					string nodeID = inputDevicePair.first;
					logging::log_log("SupraManager: Freezing input '", nodeID, "'");
					inputDevicePair.second->freeze();
				}
			}
		}
	}

	void supra::SupraManager::unfreezeInputs()
	{
		resetFreezeTimeout();

		// If inputs have not been unfrozen already
		bool expected = true;
		if (m_inputsFrozen.compare_exchange_strong(expected, false))
		{
			for (auto inputDevicePair : m_inputDevices)
			{
				if (inputDevicePair.second->getRunning())
				{
					string nodeID = inputDevicePair.first;
					logging::log_log("SupraManager: Unfreezing input '", nodeID, "'");
					inputDevicePair.second->unfreeze();
				}
			}
		}
	}

	bool SupraManager::inputsFrozen()
	{
		return m_inputsFrozen;
	}

	void SupraManager::freezeThread()
	{
		while (m_freezeThreadContinue)
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			switch(m_freezeTimeout.fetch_sub(1))
			{
				case 1:
					freezeInputs();
					break;
				case 0:
					m_freezeTimeout = 0;
					break;
				default:
					break;
			}
		}
	}

	std::string SupraManager::findUnusedID(std::string prefix)
	{
		std::vector<std::string> usedIDs(m_nodes.size());
		std::transform(m_nodes.begin(), m_nodes.end(), usedIDs.begin(),
			[](std::pair<std::string, std::shared_ptr<AbstractNode> > p) { return p.first; });

		string newID = prefix + std::to_string(rand());
		for (size_t i = 1; i <= 100; i++)
		{
			string potentialID = prefix + "_" + std::to_string(i);
			if (m_nodes.count(potentialID) == 0)
			{
				newID = potentialID;
				break;
			}
		}
		return newID;
	}

	int32_t SupraManager::getFreezeTimeout()
	{
		return m_freezeTimeout;
	}

	void SupraManager::setFreezeTimeout(int32_t timeout)
	{
		m_freezeTimeoutInit = timeout;
		resetFreezeTimeout();
	}

	int32_t SupraManager::resetFreezeTimeout()
	{
		int32_t timeoutInit = m_freezeTimeoutInit;
		m_freezeTimeout = timeoutInit;
		return m_freezeTimeout;
	}

	void SupraManager::waitForGraph()
	{
		m_graph->wait_for_all();
	}

	void SupraManager::quit()
	{
		if (m_quitCallback)
		{
			m_quitCallback();
		}
	}

	void SupraManager::setQuitCallback(std::function<void(void)> quitCallback)
	{
		m_quitCallback = quitCallback;
	}
}
