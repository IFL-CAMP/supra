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

#ifndef __RXSAMPLEBEAMFORMERTESTSIGNAL_H__
#define __RXSAMPLEBEAMFORMERTESTSIGNAL_H__

#include "USImageProperties.h"
#include "WindowFunction.h"
#include "RxBeamformerCommon.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra
{
	class RxSampleBeamformerTestSignal
	{
	public:
		template <bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
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
			const WindowFunctionGpu* windowFunction,
			const WindowFunction::ElementType* functionShared
		)
		{
			constexpr float cylinderSpacing = 6; //[mm]
			constexpr float cylinderDiameter = 2; //[mm]
			constexpr float cylinderDepth = 30; //[mm]
			constexpr int numCylindersHalf = 3;

			float sample = 0.0f;

			vec3f point{ scanline_x + dirX*depth, dirY*depth, scanline_z + dirZ*depth };
			point = point *dt*speedOfSound; // bring point position back from dt space to world space
			//check for all cylinders
			// cylinders along z axis
			for (int cylinderNo = -numCylindersHalf; cylinderNo <= numCylindersHalf; cylinderNo++)
			{
				vec3f cylinderCenter = vec3f{ cylinderNo * cylinderSpacing, cylinderDepth, 0 };
				vec3f pointInPlane = point;
				pointInPlane.z = 0;
				float distance = norm(pointInPlane - cylinderCenter);
				if (distance <= cylinderDiameter)
				{
					sample = 1000;
				}
			}
			// cylinders along x axis
			for (int cylinderNo = -numCylindersHalf; cylinderNo <= numCylindersHalf; cylinderNo++)
			{
				vec3f cylinderCenter = vec3f{ 0, cylinderDepth,  cylinderNo * cylinderSpacing };
				vec3f pointInPlane = point;
				pointInPlane.x = 0;
				float distance = norm(pointInPlane - cylinderCenter);
				if (distance <= cylinderDiameter)
				{
					sample = 1000;
				}
			}

			return sample;
		}

		template <bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
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
			const WindowFunctionGpu* windowFunction
		)
		{
			constexpr float cylinderSpacing = 6; //[mm]
			constexpr float cylinderDiameter = 2; //[mm]
			constexpr float cylinderDepth = 30; //[mm]
			constexpr int numCylindersHalf = 3;

			float sample = 0.0f;

			vec3f point{ scanline_x + dirX*depth, dirY*depth, dirZ*depth };
			point = point *dt*speedOfSound; // bring point position back from dt space to world space
			//check for all cylinders
			// cylinders along z axis
			for (int cylinderNo = -numCylindersHalf; cylinderNo <= numCylindersHalf; cylinderNo++)
			{
				vec3f cylinderCenter = vec3f{ cylinderNo * cylinderSpacing, cylinderDepth, 0 };
				vec3f pointInPlane = point;
				pointInPlane.z = 0;
				float distance = norm(pointInPlane - cylinderCenter);
				if (distance <= cylinderDiameter)
				{
					sample = 1000;
				}
			}
			return sample;
		}
	};
}

#endif //!__RXSAMPLEBEAMFORMERTESTSIGNAL_H__
