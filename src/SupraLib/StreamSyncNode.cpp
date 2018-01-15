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
			shared_ptr<USImage<int16_t> >   pInImageInt16 = dynamic_pointer_cast<USImage<int16_t>>(inObj);
			shared_ptr<USImage<uint8_t> >   pInImageUint8 = dynamic_pointer_cast<USImage<uint8_t>>(inObj);
			shared_ptr<USRawData<int16_t> > pInImageRaw = dynamic_pointer_cast<USRawData<int16_t>>(inObj);
			if (pInImageInt16)
			{
				cudaSafeCall(cudaStreamSynchronize(pInImageInt16->getData()->getStream()));
			}
			else if (pInImageUint8)
			{
				cudaSafeCall(cudaStreamSynchronize(pInImageUint8->getData()->getStream()));
			}
			else if (pInImageRaw)
			{
				cudaSafeCall(cudaStreamSynchronize(pInImageRaw->getData()->getStream()));
			}
		}
#endif
		m_callFrequency.measureEnd();
		return inObj;
	}
}