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

#ifndef __RXBEAMFORMERPCGMV_H__
#define __RXBEAMFORMERPCGMV_H__

#ifdef HAVE_BEAMFORMER_MINIMUM_VARIANCE

#include <memory>

#include <cublas_v2.h>

namespace supra
{
	class USRawData;
	class USImage;

	namespace RxBeamformerMVpcg
	{
		template <typename ChannelDataType, typename ImageDataType>
		std::shared_ptr<USImage> performRxBeamforming(
			std::shared_ptr<const USRawData> rawData,
			uint32_t subArraySize,
			uint32_t temporalSmoothing,
			uint32_t maxIterations,
			double convergenceThreshold,
			double outputClamp);
	}
}

#endif //HAVE_BEAMFORMER_MINIMUM_VARIANCE

#endif //!__RXBEAMFORMERPCGMV_H__
