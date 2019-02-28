// ================================================================================================
// 
// Copyright (C) 2017, Rüdiger Göbl - all rights reserved
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
//          Rüdiger Göbl
//          Email r.goebl@tum.de
//          Chair for Computer Aided Medical Procedures
//          Technische Universität München
//          Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License, version 2.1, as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this program.  If not, see
// <http://www.gnu.org/licenses/>.
//
// ================================================================================================

#ifndef __RXBEAMFORMERPCGMV_H__
#define __RXBEAMFORMERPCGMV_H__

#ifdef HAVE_BEAMFORMER_MINIMUM_VARIANCE

#include <memory>

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
			uint32_t maxIterationsOverride,
			double convergenceThreshold,
			double subArrayScalingPower,
			double outputClamp);
	}
}

#endif //HAVE_BEAMFORMER_MINIMUM_VARIANCE

#endif //!__RXBEAMFORMERPCGMV_H__
