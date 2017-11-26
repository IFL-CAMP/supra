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

#include "ContainerFactory.h"

#include <utilities/Logging.h>
using namespace std;

namespace supra
{
	ContainerFactory::ContainerStreamType ContainerFactory::getNextStream()
	{
		std::lock_guard<std::mutex> streamLock(sm_streamMutex);

		if (sm_streams.size() == 0)
		{
			initStreams();
		}

		size_t streamIndex = sm_streamIndex;
		sm_streamIndex = (sm_streamIndex + 1) % sm_numberStreams;
		return sm_streams[streamIndex];
	}
	void ContainerFactory::initStreams()
	{
		logging::log_log("ContainerFactory: Initializing ", sm_numberStreams, " streams.");
		sm_streamIndex = 0;
#ifdef HAVE_CUDA
		sm_streams.resize(sm_numberStreams);
		for (size_t k = 0; k < sm_numberStreams; k++)
		{
			cudaSafeCall(cudaStreamCreateWithFlags(&(sm_streams[k]), cudaStreamNonBlocking));
		}
#else
		sm_streams.resize(sm_numberStreams, 0);
#endif
	}

	std::vector<ContainerFactory::ContainerStreamType> ContainerFactory::sm_streams = {};
	size_t ContainerFactory::sm_streamIndex = 0;
	std::mutex ContainerFactory::sm_streamMutex;
}