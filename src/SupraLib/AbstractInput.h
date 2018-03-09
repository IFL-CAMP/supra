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

#ifndef __ABSTRACTINPUT_H__
#define __ABSTRACTINPUT_H__

#include <memory>
#include <thread>
#include <atomic>
#include <tuple>
#include <array>

#include <tbb/flow_graph.h>

#include "AbstractNode.h"
#include "utilities/SingleThreadTimer.h"
#include "RecordObject.h"

namespace supra
{
	/**
	 * The base class for all input nodes.
	 *
	 * An input node has no input ports on its own, but maintains at least one output port.
	 * The input nodes are the source of the dataflow. As such they are not restricted in
	 * when they are active. 
	 * There are three major threading modes for input nodes. These are described schematically
	 * in sequence diagrams in the Visual Studio models in doc/models_supra.
	 * 1. Callback driven
	 *		If the implementation receives the data via callbacks from some other interface,
	 *		it can simply put the data into the dataflow graph (\see addData) from within the 
	 *		callback
	 * 2. Polling
	 *		Once the input node is started (\see startAcquisition), the implementation is run in
	 *		its own thread. As such it can poll an device for new datasets. Yielding that thread
	 *		via sleeping can also be performed as deemed fit.
	 * 3. Timer based
	 *		For simple inputs a timer-callback can be sufficient. For that a timer is implemented
	 *		within input nodes (\see SingleThreadTimer). It is configured with the method 
	 *		\see setupTimer and once \see timerLoop is entered, the timer continously calls 
	 *		\see timerCallback for the node, sleeping appropriately inbetween calls.
	 */
	class AbstractInput : public AbstractNode
	{
	public:
		/// Base constructor for the input node. Initializes its output ports.
		AbstractInput(tbb::flow::graph& graph, const std::string & nodeID, size_t numInputs) 
			: AbstractNode(nodeID)
		{
			m_pOutputNodes.resize(numInputs);
			for (size_t i = 0; i < numInputs; i++)
			{
				m_pOutputNodes[i] = std::unique_ptr<tbb::flow::broadcast_node<std::shared_ptr<RecordObject> > >(
					new tbb::flow::broadcast_node<std::shared_ptr<RecordObject> >(graph));
			}
		}
		~AbstractInput()
		{
			waitForFinish();
		}

		/// Waits unitl the run-thread of the input node ends.
		/// As input nodes are not driven by the dataflow, we need to stop the input nodes 
		/// and then weit for them to finish before we can stop the dataflow graph.
		void waitForFinish()
		{
			if (m_pInputDeviceThread && m_pInputDeviceThread->joinable())
			{
				m_pInputDeviceThread->join();
			}
		}

		/// This method starts the input nodes by starting their separate work-threads.
		/// The entry point for the input object thread
		void start()
		{
			setRunning(true);
			//The input device function is run in a new thread, so it is free to consume time and yield
			// e.g. using the builtin timer
			m_pInputDeviceThread = std::make_shared<std::thread>(std::thread([this]() {this->startAcquisition(); }));
		}

		/// Set the state of the input node, if newState is false, the node is stopped
		virtual bool setRunning(bool newState)
		{
			bool oldState = m_running;
			m_running = newState;
			if (!m_running)
			{
				stopAcquisition();
			}
			return (oldState || newState) && !(oldState && oldState);
		}

		/// Returns whether the node is running
		bool getRunning()
		{
			return m_running;
		}

		/// returns the output port with the given index
		template <size_t index>
		tbb::flow::broadcast_node<std::shared_ptr<RecordObject> >&
			getOutputNode() {
			return *std::get<index>(m_pOutputNodes);
		}

		/// returns the number of input ports of this node.
		/// Always 0 as an input has no inputs from the dataflow graph
		virtual size_t getNumInputs() { return 0; }
		/// returns the number of output ports of this node
		virtual size_t getNumOutputs() { return m_pOutputNodes.size(); }

		/// returns a pointer to the output port with the given index
		virtual tbb::flow::sender<std::shared_ptr<RecordObject> > * getOutput(size_t index) {
			if (index < m_pOutputNodes.size())
			{
				return m_pOutputNodes[index].get();
			}
			return nullptr;
		}

	protected:
		/// The nodes output. An implementing node calls this method when it has a dataset 
		/// to send into the graph.
		template <size_t index>
		bool addData(std::shared_ptr<RecordObject> data)
		{
			return m_pOutputNodes[index]->try_put(data);
		}

		/// Returns the configured frequency of the \see SingleThreadTimer
		double getTimerFrequency()
		{
			return m_timer.getFrequency();
		}
		/// Configures the callback frequency of the \see SingleThreadTimer
		void setUpTimer(double frequency)
		{
			m_timer.setFrequency(frequency);
		}
		/// The main loop of the \see SingleThreadTimer
		/// once entered, the timer continously calls \see timerCallback for the node, 
		/// sleeping appropriately inbetween calls, until the callback returns false.
		void timerLoop()
		{
			bool shouldContinue = true;
			while (shouldContinue)
			{
				shouldContinue = timerCallback();
				if (shouldContinue) {
					m_timer.sleepUntilNextSlot();
				}
			}
		}
	private:
		
		std::vector<std::unique_ptr<tbb::flow::broadcast_node<std::shared_ptr<RecordObject> > > > m_pOutputNodes;

		SingleThreadTimer m_timer;
		std::shared_ptr<std::thread> m_pInputDeviceThread;
		std::atomic_bool m_running;

		size_t m_numInputs;

		//Functions to be overwritten
	public:
		/// The main intialization of the input node should happen in this method.
		/// It is called only after the initial configuration is applied.
		/// Needs to be overwritten by the implementing node.
		virtual void initializeDevice() {};
		/// Implementations signal whether they are ready to be started using this method.
		/// Needs to be overwritten by the implementing node.
		virtual bool ready() { return false; };

		/// returns a vector with the indices of the output ports, that provide image data
		/// This is mostly used for the purpose of visualization
		virtual std::vector<size_t> getImageOutputPorts() = 0;
		/// returns a vector with the indices of the output ports, that provide tracking data
		/// This is mostly used for the purpose of visualization
		virtual std::vector<size_t> getTrackingOutputPorts() = 0;

		virtual void freeze() = 0;
		virtual void unfreeze() = 0;
	protected:
		/// The entry point for the implementing input node
		/// This method is called in a separate thread once the node is started.
		virtual void startAcquisition() = 0;
		/// This method is called when the input node is to be stopped.
		/// As it is asynchronous to the node's working thread, it need to be thread safe
		/// and should bring the node's main function to a stop.
		virtual void stopAcquisition() {};
		////Needs to be thread safe
		//virtual void configurationEntryChanged(const std::string& configKey) = 0;
		////Needs to be thread safe
		//virtual void configurationChanged() = 0;

		/// The timer callback for input nodes that use the \see SingleThreadTimer for 
		/// triggering itself. This method is called repeatedly and with the configured rate
		/// once timerLoop has been entered and as long as the callback returns true.
		virtual bool timerCallback() { return false; };
	};
}
#endif //!__ABSTRACTINPUT_H__
