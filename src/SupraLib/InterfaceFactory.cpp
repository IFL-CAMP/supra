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

#include "InterfaceFactory.h"

#include "utilities/Logging.h"
#include "InputOutput/TrackerInterfaceSimulated.h"
#include "InputOutput/TrackerInterfaceIGTL.h"
#include "InputOutput/TrackerInterfaceROS.h"
#include "InputOutput/UltrasoundInterfaceSimulated.h"
#include "InputOutput/UltrasoundInterfaceUltrasonix.h"
#include "InputOutput/OpenIGTLinkOutputDevice.h"
#include "InputOutput/UsIntCephasonicsBmode.h"
#include "InputOutput/UsIntCephasonicsBtcc.h"
#include "InputOutput/UsIntCephasonicsCc.h"
#include "InputOutput/MetaImageOutputDevice.h"
#include "InputOutput/RosImageOutputDevice.h"
#include "InputOutput/UltrasoundInterfaceRawDataMock.h"
#include "Beamformer/BeamformingNode.h"
#include "Beamformer/IQDemodulatorNode.h"
#include "Beamformer/HilbertEnvelopeNode.h"
#include "Beamformer/LogCompressorNode.h"
#include "Beamformer/ScanConverterNode.h"
#include "Beamformer/TemporalFilterNode.h"
#include "Beamformer/RawDelayNode.h"
#include "StreamSyncNode.h"
#include "TemporalOffsetNode.h"
#include "StreamSynchronizer.h"
#include "FrequencyLimiterNode.h"
#include "AutoQuitNode.h"
#include "ExampleNodes/ImageProcessingCpuNode.h"
#include "ExampleNodes/ImageProcessingCudaNode.h"

using namespace std;

namespace supra
{
	shared_ptr<tbb::flow::graph> InterfaceFactory::createGraph()
	{
		new tbb::flow::graph();
		return make_shared<tbb::flow::graph>();
	}

	shared_ptr<AbstractInput> InterfaceFactory::createInputDevice(shared_ptr<tbb::flow::graph> pG, const std::string & nodeID, std::string deviceType, size_t numPorts)
	{
		shared_ptr<AbstractInput> retVal = shared_ptr<AbstractInput>(nullptr);

		if (numPorts>1 && deviceType != "UltrasoundInterfaceCephasonicsCC")
		{
			logging::log_warn("InterfaceFactory: More than one port currently not supported for input device " + deviceType + ".");
		}

#ifdef HAVE_DEVICE_TRACKING_SIM
		if (deviceType == "TrackerInterfaceSimulated")
		{
			retVal = make_shared<TrackerInterfaceSimulated>(*pG, nodeID);
		}
#endif //HAVE_DEVICE_TRACKING_SIM
#ifdef HAVE_DEVICE_TRACKING_IGTL
		if (deviceType == "TrackerInterfaceIGTL")
		{
			retVal = make_shared<TrackerInterfaceIGTL>(*pG, nodeID);
		}
#endif //HAVE_DEVICE_TRACKING_IGTL
#ifdef HAVE_DEVICE_TRACKING_ROS
		if (deviceType == "TrackerInterfaceROS")
		{
			retVal = make_shared<TrackerInterfaceROS>(*pG, nodeID);
		}
#endif //HAVE_DEVICE_TRACKING_IGTL
#ifdef HAVE_DEVICE_ULTRASOUND_SIM
		if (deviceType == "UltrasoundInterfaceSimulated")
		{
			retVal = make_shared<UltrasoundInterfaceSimulated>(*pG, nodeID);
		}
#endif //HAVE_DEVICE_ULTRASOUND_SIM
#ifdef HAVE_DEVICE_ULTRASONIX
		if (deviceType == "UltrasoundInterfaceUltrasonix")
		{
			retVal = make_shared<UltrasoundInterfaceUltrasonix>(*pG, nodeID);
		}
#endif //HAVE_DEVICE_ULTRASONIX
#ifdef HAVE_DEVICE_CEPHASONICS
		if (deviceType == "UltrasoundInterfaceCephasonics")
		{
			retVal = make_shared<UsIntCephasonicsBmode>(*pG, nodeID);
		}
		if (deviceType == "UltrasoundInterfaceCephasonicsBTCC")
		{
			retVal = make_shared<UsIntCephasonicsBtcc>(*pG, nodeID);
		}
#ifdef HAVE_CUDA
		if (deviceType == "UltrasoundInterfaceCephasonicsCC")
		{
			retVal = make_shared<UsIntCephasonicsCc>(*pG, nodeID, numPorts);
		}
#endif //HAVE_CUDA
#endif //HAVE_DEVICE_CEPHASONICS
#ifdef HAVE_BEAMFORMER
		if (deviceType == "UltrasoundInterfaceRawDataMock")
		{
			retVal = make_shared<UltrasoundInterfaceRawDataMock>(*pG, nodeID);
		}
#endif

		logging::log_error_if(!((bool)retVal),
			"Error creating input device. Requested type '", deviceType, "' is unknown. Did you activate the corresponding module in the build of the library?");
		logging::log_info_if((bool)retVal,
			"Created input device '", deviceType, "' with ID '", nodeID, "'");
		return retVal;
	}

	shared_ptr<AbstractOutput> InterfaceFactory::createOutputDevice(shared_ptr<tbb::flow::graph> pG, const std::string & nodeID, std::string deviceType, bool queueing)
	{
		shared_ptr<AbstractOutput> retVal = shared_ptr<AbstractOutput>(nullptr);
#ifdef HAVE_DEVICE_IGTL_OUTPUT
		if (deviceType == "OpenIGTLinkOutputDevice")
		{
			retVal = make_shared<OpenIGTLinkOutputDevice>(*pG, nodeID, queueing);
		}
#endif //HAVE_DEVICE_IGTL_OUTPUT
#ifdef HAVE_DEVICE_METAIMAGE_OUTPUT
		if (deviceType == "MetaImageOutputDevice")
		{
			retVal = make_shared<MetaImageOutputDevice>(*pG, nodeID, queueing);
		}
#endif //HAVE_DEVICE_METAIMAGE_OUTPUT
#ifdef HAVE_DEVICE_ROSIMAGE_OUTPUT
		if (deviceType == "RosImageOutputDevice")
		{
			retVal = make_shared<RosImageOutputDevice>(*pG, nodeID, queueing);
		}
#endif //HAVE_DEVICE_ROSIMAGE_OUTPUT
		logging::log_error_if(!((bool)retVal),
			"Error creating output device. Requested type '", deviceType, "' is unknown. Did you activate the corresponding module in the build of the library?");
		logging::log_info_if((bool)retVal,
			"Created output device '", deviceType, "' with ID '", nodeID, "'");
		return retVal;
	}

	shared_ptr<AbstractNode> InterfaceFactory::createNode(shared_ptr<tbb::flow::graph> pG, const std::string & nodeID, std::string nodeType, bool queueing)
	{
		shared_ptr<AbstractNode> retVal = shared_ptr<AbstractNode>(nullptr);

		if (m_nodeCreators.count(nodeType) != 0)
		{
			retVal = m_nodeCreators[nodeType](*pG, nodeID, queueing);
		}

		logging::log_error_if(!((bool)retVal),
			"Error creating node. Requested type '", nodeType, "' is unknown. Did you activate the corresponding module in the build of the library?");
		logging::log_info_if((bool)retVal,
			"Created node '", nodeType, "' with ID '", nodeID, "'");
		return retVal;
	}

	std::vector<std::string> InterfaceFactory::getNodeTypes()
	{
		std::vector<std::string> nodeTypes(m_nodeCreators.size());
		std::transform(m_nodeCreators.begin(), m_nodeCreators.end(), nodeTypes.begin(), 
			[](std::pair<std::string, nodeCreationFunctionType> entry) { return entry.first; });
		return nodeTypes;
	}

	std::map<std::string, InterfaceFactory::nodeCreationFunctionType> 
		InterfaceFactory::m_nodeCreators = 
	{
		{ "StreamSynchronizer",     [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<StreamSynchronizer>(g, nodeID, queueing); } },
		{ "TemporalOffsetNode",     [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<TemporalOffsetNode>(g, nodeID, queueing); } },
		{ "FrequencyLimiterNode",   [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<FrequencyLimiterNode>(g, nodeID, queueing); } },
		{ "AutoQuitNode",           [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<AutoQuitNode>(g, nodeID, queueing); } },
		{ "StreamSyncNode",         [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<StreamSyncNode>(g, nodeID, queueing); } },
		{ "ImageProcessingCpuNode", [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<ImageProcessingCpuNode>(g, nodeID, queueing); } },
#ifdef HAVE_CUDA
		{ "ImageProcessingCudaNode", [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<ImageProcessingCudaNode>(g, nodeID, queueing); } },
#endif
#ifdef HAVE_CUFFT
		{ "HilbertEnvelopeNode", [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<HilbertEnvelopeNode>(g, nodeID, queueing); } },
#endif
#ifdef HAVE_BEAMFORMER
		{ "BeamformingNode",    [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<BeamformingNode>(g, nodeID, queueing); } },
		{ "IQDemodulatorNode",  [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<IQDemodulatorNode>(g, nodeID, queueing); } },
		{ "LogCompressorNode",  [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<LogCompressorNode>(g, nodeID, queueing); } },
		{ "ScanConverterNode",  [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<ScanConverterNode>(g, nodeID, queueing); } },
		{ "TemporalFilterNode", [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<TemporalFilterNode>(g, nodeID, queueing); } },
		{ "RawDelayNode",       [](tbb::flow::graph& g, std::string nodeID, bool queueing) { return make_shared<RawDelayNode>(g, nodeID, queueing); } },
#endif
	};
}
