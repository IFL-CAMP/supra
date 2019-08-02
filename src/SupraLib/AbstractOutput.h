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

#ifndef __ABSTRACTOUTPUT_H__
#define __ABSTRACTOUTPUT_H__

#include <memory>
#include <atomic>

#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "ConfigurationDictionary.h"
#include "RecordObject.h"

namespace supra
{
	/*! \brief Abstract interface for output nodes. 
	*		   Data arrives through the TBB node and is processed in writeData.
	*
	*  This abstract base class contains the Thread Building Blocks node, that provides the input for the
	*  implementing classes.
	*/
	class AbstractOutput : public AbstractNode
	{
	public:
		/// Base constructor for output nodes
		AbstractOutput(tbb::flow::graph& graph, const std::string & nodeID, bool queueing)
			: AbstractNode(nodeID, queueing)
		{ 	
			if (queueing)
			{
				m_inputNode = std::unique_ptr<NodeTypeOneSidedQueueing>(
					new NodeTypeOneSidedQueueing(graph, 1,
						[this](const std::shared_ptr<RecordObject> & inMessage) {
						if (this->m_running)
						{
							writeData(inMessage);
						}
					})
				);
			}
			else
			{
				m_inputNode = std::unique_ptr<NodeTypeOneSidedDiscarding>(
					new NodeTypeOneSidedDiscarding(graph, 1,
						[this](const std::shared_ptr<RecordObject> & inMessage) {
						if (this->m_running)
						{
							writeData(inMessage);
						}
					})
				);
			}
		}

		/// Set the state of the output node, if newState is false, the node is stopped
		virtual bool setRunning(bool newState)
		{
			bool oldState = m_running;
			m_running = newState;
			if (!m_running)
			{
				stopOutput();
			}
			return (oldState || newState) && !(oldState && oldState);
		}

		/// Returns whether the node is running
		bool getRunning()
		{
			return m_running;
		}

		/// returns the number of input ports of this node
		virtual size_t getNumInputs() { return 1; }
		/// returns the number of output ports of this node.
		/// Always 0 as an output has no outputs to the dataflow graph
		virtual size_t getNumOutputs() { return 0; }

		/// returns a pointer to the input port with the given index
		virtual tbb::flow::graph_node * getInput(size_t index) {
			if (index == 0)
			{
				return m_inputNode.get();
			}
			return nullptr;
		}

	protected:
		virtual void configurationEntryChanged(const std::string& configKey) 
		{
			configurationDone();
		};
		
		virtual void configurationChanged()
		{
			configurationDone();
		}

	private:
		std::unique_ptr<tbb::flow::graph_node> m_inputNode;
		std::atomic_bool m_running;

		//Functions to be overwritten
	public:
		/// The method can be overwritten by an implementing output node.
		/// It is called before the inputs are started and thus before any messages cas 
		/// arrive at any other node. It allows the implementation to perform initializations
		/// after the configuration has been performed.
		virtual void initializeOutput() {};
		/// Implementations signal whether they are ready to be started using this method.
		/// Needs to be overwritten by the implementing node.
		virtual bool ready() { return false; };
		/// Callback for the implementation, signals the start of a recording sequence.
		/// The concept of sequence allows several sequences to be recorded (through starting
		/// and stopping) within one run of output node.
		/// As it is called from a different thread than the output node runs in, 
		/// it needs to be thread safe.
		virtual void startSequence() {};
		/// Callback for the implementation, signals the end of a recording sequence.
		/// The concept of sequence allows several sequences to be recorded (through starting
		/// and stopping) within one run of output node.
		/// As it is called from a different thread than the output node runs in, 
		/// it needs to be thread safe.
		virtual void stopSequence() {};
	protected:
		/// This method is called for first start the output node
		virtual void startOutput() {};
		/// This method is called completly stop the output node.
		/// As it is called from a different thread than the output node runs in, 
		/// it needs to be thread safe.
		virtual void stopOutput() {};
		/// This method is called after the initial configuration of the node.
		/// As it is called from a different thread than the output node runs in, 
		/// it needs to be thread safe.
		virtual void configurationDone() = 0;

		/// The main part of all output nodes. 
		/// Here the data is output to the corresponding stream/file/socket.
		virtual void writeData(std::shared_ptr<RecordObject> data) = 0;
	};
}

#endif //!__ABSTRACTOUTPUT_H__
