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

#ifndef __CONTAINERFACTORY_H__
#define __CONTAINERFACTORY_H__

#ifdef HAVE_CUDA
#include "utilities/cudaUtility.h"
#endif
#include "Container.h"

#include <vector>
#include <mutex>

namespace supra
{
	class ContainerFactory
	{
	public:
		typedef Container<int>::ContainerStreamType ContainerStreamType;

		static ContainerStreamType getNextStream();

	private:
		static void initStreams();

		static constexpr size_t sm_numberStreams = 16;

		static std::vector<ContainerStreamType> sm_streams;
		static size_t sm_streamIndex;
		static std::mutex sm_streamMutex;
	};
}

#endif //!__CONTAINERFACTORY_H__
