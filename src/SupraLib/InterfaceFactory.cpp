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

	shared_ptr<AbstractInput<RecordObject>> InterfaceFactory::createInputDevice(shared_ptr<tbb::flow::graph> pG, const std::string & nodeID, std::string deviceType)
	{
		shared_ptr<AbstractInput<RecordObject>> retVal = shared_ptr<AbstractInput<RecordObject> >(nullptr);

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
#ifdef HAVE_DEVICE_METAIMAGE_OUTPUT
		if (deviceType == "MetaImageOutputDevice")
		{
			retVal = make_shared<MetaImageOutputDevice>(*pG, nodeID, queueing);
		}
#endif //HAVE_DEVICE_METAIMAGE_OUTPUT
		logging::log_error_if(!((bool)retVal),
			"Error creating output device. Requested type '", deviceType, "' is unknown. Did you activate the corresponding module in the build of the library?");
		logging::log_info_if((bool)retVal,
			"Created output device '", deviceType, "' with ID '", nodeID, "'");
		return retVal;
	}

	shared_ptr<AbstractNode> InterfaceFactory::createNode(shared_ptr<tbb::flow::graph> pG, const std::string & nodeID, std::string nodeType, bool queueing)
	{
		shared_ptr<AbstractNode> retVal = shared_ptr<AbstractNode>(nullptr);

		if (nodeType == "StreamSynchronizer")
		{
			retVal = make_shared<StreamSynchronizer>(*pG, nodeID, queueing);
		}
		if (nodeType == "TemporalOffsetNode")
		{
			retVal = make_shared<TemporalOffsetNode>(*pG, nodeID, queueing);
		}
		if (nodeType == "FrequencyLimiterNode")
		{
			retVal = make_shared<FrequencyLimiterNode>(*pG, nodeID, queueing);
		}
		if (nodeType == "AutoQuitNode")
		{
			retVal = make_shared<AutoQuitNode>(*pG, nodeID, queueing);
		}
		if (nodeType == "StreamSyncNode")
		{
			retVal = make_shared<StreamSyncNode>(*pG, nodeID, queueing);
		}
		if (nodeType == "ImageProcessingCpuNode")
		{
			retVal = make_shared<ImageProcessingCpuNode>(*pG, nodeID, queueing);
		}
#ifdef HAVE_CUDA
		if (nodeType == "ImageProcessingCudaNode")
		{
			retVal = make_shared<ImageProcessingCudaNode>(*pG, nodeID, queueing);
		}
#endif
#ifdef HAVE_BEAMFORMER
		if (nodeType == "BeamformingNode")
		{
			retVal = make_shared<BeamformingNode>(*pG, nodeID, queueing);
		}
		if (nodeType == "IQDemodulatorNode")
		{
			retVal = make_shared<IQDemodulatorNode>(*pG, nodeID, queueing);
		}
#ifdef HAVE_CUFFT
		if (nodeType == "HilbertEnvelopeNode")
		{
			retVal = make_shared<HilbertEnvelopeNode>(*pG, nodeID, queueing);
		}
#endif
		if (nodeType == "LogCompressorNode")
		{
			retVal = make_shared<LogCompressorNode>(*pG, nodeID, queueing);
		}
		if (nodeType == "ScanConverterNode")
		{
			retVal = make_shared<ScanConverterNode>(*pG, nodeID, queueing);
		}
		if (nodeType == "TemporalFilterNode")
		{
			retVal = make_shared<TemporalFilterNode>(*pG, nodeID, queueing);
		}
		if (nodeType == "RawDelayNode")
		{
			retVal = make_shared<RawDelayNode>(*pG, nodeID, queueing);
		}
#endif
		logging::log_error_if(!((bool)retVal),
			"Error creating node. Requested type '", nodeType, "' is unknown. Did you activate the corresponding module in the build of the library?");
		logging::log_info_if((bool)retVal,
			"Created node '", nodeType, "' with ID '", nodeID, "'");
		return retVal;
	}
}
