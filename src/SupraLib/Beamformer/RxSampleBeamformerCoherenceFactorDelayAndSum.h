// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __RXSAMPLEBEAMFORMERCOHERENCEFACTORDELAYANDSUM_H__
#define __RXSAMPLEBEAMFORMERCOHERENCEFACTORDELAYANDSUM_H__

#include "USImageProperties.h"
#include "WindowFunction.h"
#include "RxBeamformerCommon.h"

#include "RxSampleBeamformerDelayAndSum.h"

// Beamformer accoring to
//

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra
{
	class RxSampleBeamformerCoherenceFactorDelayAndSum
	{
	public:
		template <bool interpolateRFlines, bool nonlinearElementToChannelMapping, typename RFType, typename ResultType, typename LocationType>
		static __device__ ResultType sampleBeamform3D(
			ScanlineRxParameters3D::TransmitParameters txParams,
			const RFType* RF,
			vec2T<uint32_t> elementLayout,
			uint32_t numReceivedChannels,
			uint32_t numTimesteps,
			const LocationType* x_elemsDTsh,
			const LocationType* z_elemsDTsh,
			LocationType scanline_x,
			LocationType scanline_z,
			LocationType dirX,
			LocationType dirY,
			LocationType dirZ,
			LocationType aDT,
			LocationType depth,
			vec2f invMaxElementDistance,
			LocationType speedOfSound,
			LocationType dt,
			int32_t additionalOffset,
			const WindowFunctionGpu* windowFunction,
			const WindowFunction::ElementType* functionShared,
			const int32_t* elementToChannelMap
		)
		{
			float value = 0.0f;
			float coherentSum = 0.0f;
			float totalEnergy = 0.0f;
			int numAdds = 0;
			LocationType initialDelay = txParams.initialDelay;
			uint32_t txScanlineIdx = txParams.txScanlineIdx;

			ResultType resultDas = RxSampleBeamformerDelayAndSum::sampleBeamform3D<interpolateRFlines, nonlinearElementToChannelMapping, RFType, ResultType, LocationType>(
				txParams,
				RF,
				elementLayout,
				numReceivedChannels,
				numTimesteps,
				x_elemsDTsh,
				z_elemsDTsh,
				scanline_x,
				scanline_z,
				dirX,
				dirY,
				dirZ,
				aDT,
				depth,
				invMaxElementDistance,
				speedOfSound,
				dt,
				additionalOffset,
				windowFunction,
				functionShared,
				elementToChannelMap);

			for (uint32_t elemIdxX = txParams.firstActiveElementIndex.x; elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++)
			{
				for (uint32_t elemIdxY = txParams.firstActiveElementIndex.y; elemIdxY < txParams.lastActiveElementIndex.y; elemIdxY++)
				{
					uint32_t elemIdx = elemIdxX + elemIdxY*elementLayout.x;
					uint32_t  channelIdx;
					if (nonlinearElementToChannelMapping)
					{
						if (elementToChannelMap[elemIdx] == USTransducer::ElementChannelMapNotConnected)
						{
							// This element was not connected to any of the channels. Nothing to do for it.
							continue;
						}
						channelIdx = elementToChannelMap[elemIdx];
					}
					else
					{
						channelIdx = elemIdx % numReceivedChannels;
					}
					LocationType x_elem = x_elemsDTsh[elemIdx];
					LocationType z_elem = z_elemsDTsh[elemIdx];

					if ((squ(x_elem - scanline_x) + squ(z_elem - scanline_z)) <= aDT)
					{
						numAdds++;
						if (interpolateRFlines)
						{
							LocationType delayf = initialDelay +
								computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x, scanline_z, depth) + additionalOffset;
							uint32_t delay = static_cast<uint32_t>(::floor(delayf));
							delayf -= delay;
							if (delay < (numTimesteps - 1))
							{
								value = 
									((1.0f - delayf) * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps] +
										delayf  * RF[(delay + 1) + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps]);
							}
							else if (delay < numTimesteps && delayf == 0.0)
							{
								value = RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
							}
						}
						else
						{
							uint32_t delay = static_cast<uint32_t>(::round(
								initialDelay + computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x, scanline_z, depth)) + additionalOffset);
							if (delay < numTimesteps)
							{
								value = RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
							}
						}
						coherentSum += value;
						totalEnergy += squ(value);
					}
				}
			}
			if (numAdds > 0)
			{
				return squ(coherentSum) / totalEnergy / numAdds * resultDas;
			}
			else
			{
				return 0;
			}
		}

		template <bool interpolateRFlines, bool nonlinearElementToChannelMapping, typename RFType, typename ResultType, typename LocationType>
		static __device__ ResultType sampleBeamform2D(
			ScanlineRxParameters3D::TransmitParameters txParams,
			const RFType* RF,
			uint32_t numTransducerElements,
			uint32_t numReceivedChannels,
			uint32_t numTimesteps,
			const LocationType* x_elemsDT,
			LocationType scanline_x,
			LocationType dirX,
			LocationType dirY,
			LocationType dirZ,
			LocationType aDT,
			LocationType depth,
			LocationType invMaxElementDistance,
			LocationType speedOfSound,
			LocationType dt,
			int32_t additionalOffset,
			const WindowFunctionGpu* windowFunction,
			const int32_t* elementToChannelMap
		)
		{
			float value = 0.0f;
			float coherentSum = 0.0f;
			float totalEnergy = 0.0f;
			int numAdds = 0;
			LocationType initialDelay = txParams.initialDelay;
			uint32_t txScanlineIdx = txParams.txScanlineIdx;

			ResultType resultDas = RxSampleBeamformerDelayAndSum::sampleBeamform2D<interpolateRFlines, nonlinearElementToChannelMapping, RFType, ResultType, LocationType>(
				txParams,
				RF,
				numTransducerElements,
				numReceivedChannels,
				numTimesteps,
				x_elemsDT,
				scanline_x,
				dirX,
				dirY,
				dirZ,
				aDT,
				depth,
				invMaxElementDistance,
				speedOfSound,
				dt,
				additionalOffset,
				windowFunction,
				elementToChannelMap);

			for (int32_t elemIdxX = txParams.firstActiveElementIndex.x; elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++)
			{
				uint32_t channelIdx;
				if (nonlinearElementToChannelMapping)
				{
					if (elementToChannelMap[elemIdxX] == USTransducer::ElementChannelMapNotConnected)
					{
						// This element was not connected to any of the channels. Nothing to do for it.
						continue;
					}
					channelIdx = elementToChannelMap[elemIdxX];
				}
				else
				{
					channelIdx = elemIdxX % numReceivedChannels;
				}
				LocationType x_elem = x_elemsDT[elemIdxX];
				if (abs(x_elem - scanline_x) <= aDT)
				{
					numAdds++;
					if (interpolateRFlines)
					{
						LocationType delayf = initialDelay +
							computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth) + additionalOffset;
						int32_t delay = static_cast<int32_t>(floor(delayf));
						delayf -= delay;
						if (delay < (numTimesteps - 1))
						{
							value =
								((1.0f - delayf) * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps] +
									delayf  * RF[(delay + 1) + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps]);
						}
						else if (delay < numTimesteps && delayf == 0.0)
						{
							value = RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
						}
					}
					else
					{
						int32_t delay = static_cast<int32_t>(round(
							initialDelay + computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth)) + additionalOffset);
						if (delay < numTimesteps)
						{
							value = RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
						}
					}
					coherentSum += value;
					totalEnergy += squ(value);
				}
			}
			if (numAdds > 0)
			{
				return squ(coherentSum) / totalEnergy / numAdds * resultDas;
			}
			else
			{
				return 0;
			}
		}
	};
}

#endif //!__RXSAMPLEBEAMFORMERCOHERENCEFACTORDELAYANDSUM_H__
