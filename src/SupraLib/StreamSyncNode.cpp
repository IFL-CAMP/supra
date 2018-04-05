// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2017, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "StreamSyncNode.h"

#include "USImage.h"
#include "Beamformer/USRawData.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	StreamSyncNode::StreamSyncNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_node(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndSynchronize(inObj); })
	{
		m_callFrequency.setName(nodeID);
		configurationChanged();
	}

	void StreamSyncNode::configurationChanged()
	{
	}

	void StreamSyncNode::configurationEntryChanged(const std::string& configKey)
	{
	}
	shared_ptr<RecordObject> StreamSyncNode::checkTypeAndSynchronize(shared_ptr<RecordObject> inObj)
	{
		m_callFrequency.measure();
#ifdef HAVE_CUDA
		if (inObj && (inObj->getType() == TypeUSImage || inObj->getType() == TypeUSRawData))
		{
			shared_ptr<USImage>   pInImage= dynamic_pointer_cast<USImage>(inObj);
			shared_ptr<USRawData> pInRaw = dynamic_pointer_cast<USRawData>(inObj);

			Container<int16_t>::ContainerStreamType stream;
			if (pInImage)
			{
				switch (pInImage->getDataType())
				{
				case TypeFloat:
					stream = pInImage->getData<float>()->getStream();
					break;
				case TypeInt16:
					stream = pInImage->getData<int16_t>()->getStream();
					break;
				case TypeUint8:
					stream = pInImage->getData<uint8_t>()->getStream();
					break;
				default:
					logging::log_error("StreamSyncNode: Image data type not supported");
				}
				cudaSafeCall(cudaStreamSynchronize(stream));
			}
			else if (pInRaw)
			{
				switch (pInRaw->getDataType())
				{
				case TypeFloat:
					stream = pInRaw->getData<float>()->getStream();
					break;
				case TypeInt16:
					stream = pInRaw->getData<int16_t>()->getStream();
					break;
				case TypeUint8:
					stream = pInRaw->getData<uint8_t>()->getStream();
					break;
				default:
					logging::log_error("StreamSyncNode: Image data type not supported");
				}
				cudaSafeCall(cudaStreamSynchronize(stream));
			}
		}
#endif
		m_callFrequency.measureEnd();
		return inObj;
	}
}