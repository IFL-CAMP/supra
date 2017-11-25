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
		m_freezeTimeoutInit = 5 * 60;
		m_inputsFrozen = false;
		resetFreezeTimeout();
	}

	std::shared_ptr<SupraManager> SupraManager::Get()
	{
		static std::shared_ptr<SupraManager> instance = std::shared_ptr<SupraManager>(new SupraManager());
		return instance;
	}

	void SupraManager::readFromXml(const char * configXmlFilename)
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
					readOutputDevicesFromXml(outputsElement);
				}
				//Other nodes
				XMLElement* nodesElement = devicesElement->FirstChildElement("nodes");
				if (nodesElement)
				{
					readNodesFromXml(nodesElement);
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

			//create input element
			auto in = InterfaceFactory::createInputDevice(m_graph, inputID, inputType);

			if (in)
			{
				//load config for this element
				ConfigurationDictionary dict(nextInput);
				in->changeConfig(dict);

				//store input node
				bool couldAdd = addNode(inputID, in);
				logging::log_warn_if(!couldAdd, "Node '", inputID, "' already existed. Did not add it to collection.");
				if (couldAdd)
				{
					m_inputDevices[inputID] = in;
				}
			}

			//advance to next element
			nextInput = nextInput->NextSiblingElement("input");
		}
	}

	void SupraManager::readOutputDevicesFromXml(tinyxml2::XMLElement * outputsElement)
	{
		XMLElement* nextOutput = outputsElement->FirstChildElement("output");
		while (nextOutput)
		{
			//find type and id of input element
			string outputType = nextOutput->Attribute("type");
			string outputID = nextOutput->Attribute("id");

			//create input element
			auto out = InterfaceFactory::createOutputDevice(m_graph, outputID, outputType);

			if (out)
			{
				//load config for this element
				ConfigurationDictionary dict(nextOutput);
				out->changeConfig(dict);

				//store output node
				bool couldAdd = addNode(outputID, out);
				logging::log_warn_if(!couldAdd, "Node '", outputID, "' already existed. Did not add it to collection.");
				if (couldAdd)
				{
					m_outputDevices[outputID] = out;
				}
			}

			//advance to next element
			nextOutput = nextOutput->NextSiblingElement("output");
		}
	}

	void SupraManager::readNodesFromXml(tinyxml2::XMLElement * nodesElement)
	{
		XMLElement* nextNode = nodesElement->FirstChildElement("node");
		while (nextNode)
		{
			//find type and id of node element
			string nodeType = nextNode->Attribute("type");
			string nodeID = nextNode->Attribute("id");

			//create node
			auto node = InterfaceFactory::createNode(m_graph, nodeID, nodeType);

			if (node)
			{
				//load parameters for this element
				ConfigurationDictionary dict(nextNode);
				node->changeConfig(dict);

				//store node
				bool couldAdd = addNode(nodeID, node);
				logging::log_warn_if(!couldAdd, "Node '", nodeID, "' already existed. Did not add it to collection.");
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
				logging::log_error("Error parsing the port attribute of a connection from '", fromID, "'.");
			}

			//to
			XMLElement* toElement = nextConnection->FirstChildElement("to");
			string toID = toElement->Attribute("id");
			int toPort = 0;
			if (toElement->QueryIntAttribute("port", &toPort) != XML_SUCCESS)
			{
				logging::log_error("Error parsing the port attribute of a connection to '", toID, "'.");
			}

			//create the connection
			connect(fromID, fromPort, toID, toPort);

			//advance to next element
			nextConnection = nextConnection->NextSiblingElement("connection");
		}
	}

	std::vector<std::string> SupraManager::getInputDeviceIDs()
	{
		std::vector<std::string> nodeIDs(m_inputDevices.size());
		std::transform(m_inputDevices.begin(), m_inputDevices.end(), nodeIDs.begin(),
			[](pair<string, shared_ptr<AbstractInput<RecordObject> > > mapPair) -> std::string {return mapPair.first; });
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

	shared_ptr<AbstractNode> SupraManager::getNode(string nodeID)
	{
		shared_ptr<AbstractNode> retVal = shared_ptr<AbstractNode>(nullptr);
		if (nodeExists(nodeID))
		{
			retVal = m_nodes[nodeID];
		}
		return retVal;
	}

	shared_ptr<AbstractInput<RecordObject>> SupraManager::getInputDevice(string nodeID)
	{
		shared_ptr<AbstractInput<RecordObject>> retVal = shared_ptr<AbstractInput<RecordObject>>(nullptr);
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

	bool SupraManager::addNode(string nodeID, shared_ptr<AbstractNode> node)
	{
		if (nodeExists(nodeID))
		{
			return false;
		}
		else
		{
			m_nodes[nodeID] = node;
			return true;
		}
	}

	bool SupraManager::nodeExists(string nodeID)
	{
		return (m_nodes.find(nodeID) != m_nodes.end());
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
				tbb::flow::make_edge(
					*(fromNode->getOutput(fromPort)),
					*(toNode->getInput(toPort)));
				logging::log_log("Added connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, ").");
			}
			else {
				logging::log_error("Could not add connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). One of the ports does not exist.");
			}
		}
		else {
			logging::log_error("Could not add connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). One node does not exist.");
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
				tbb::flow::remove_edge(
					*(fromNode->getOutput(fromPort)),
					*(toNode->getInput(toPort)));
				logging::log_log("Removed connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, ").");
			}
			else {
				logging::log_error("Could not remove connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). One of the ports does not exist.");
			}
		}
		else {
			logging::log_error("Could not remove connection from (", fromID, ", ", fromPort, ") to (", toID, ", ", toPort, "). One node does not exist.");
		}
	}

	void SupraManager::startOutputs()
	{
		for (auto outputDevicePair : m_outputDevices)
		{
			string nodeID = outputDevicePair.first;
			logging::log_log("Starting output '", nodeID, "'");
			outputDevicePair.second->initializeOutput();
			if (outputDevicePair.second->ready())
			{
				outputDevicePair.second->setRunning(true);
			}
			else {
				logging::log_warn("Output '", nodeID, "' not started. It was not ready.");
			}
		}
	}

	void SupraManager::startOutputsSequence()
	{
		for (auto outputDevicePair : m_outputDevices)
		{
			string nodeID = outputDevicePair.first;
			logging::log_log("Starting sequence at '", nodeID, "'");
			outputDevicePair.second->startSequence();
		}
	}

	void SupraManager::stopOutputsSequence()
	{
		for (auto outputDevicePair : m_outputDevices)
		{
			string nodeID = outputDevicePair.first;
			logging::log_log("Stopping sequence at '", nodeID, "'");
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
			logging::log_log("Starting input '", nodeID, "'");
			inputDevicePair.second->initializeDevice();
			if (inputDevicePair.second->ready())
			{
				inputDevicePair.second->start();
			}
			else {
				logging::log_warn("Input '", nodeID, "' not started. It was not ready.");
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
				logging::log_log("Stopping input '", nodeID, "'");
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
				logging::log_log("Waiting for input '", nodeID, "' to finish.");
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
					logging::log_log("Freezing input '", nodeID, "'");
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
					logging::log_log("Freezing input '", nodeID, "'");
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
