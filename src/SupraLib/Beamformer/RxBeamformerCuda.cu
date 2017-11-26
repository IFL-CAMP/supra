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
#include "RxBeamformerCuda.h"
#include "Beamformer.h"
#include "USImage.h"
#include "USRawData.h"
#include "utilities/cudaUtility.h"
#include "utilities/Logging.h"

#include <memory>
#include <cassert>

#include <iomanip>

using std::make_shared;
using std::unique_ptr;
using std::string;
using std::vector;
using std::const_pointer_cast;

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly

namespace supra
{
	RxBeamformerCuda::RxBeamformerCuda(std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > rxParameters, size_t numDepths, double depth, double speedOfSoundMMperS, const USTransducer* pTransducer)
		: m_speedOfSoundMMperS(speedOfSoundMMperS)
		, m_rxNumDepths(numDepths)
		, m_windowFunction(nullptr)
	{
		m_lastSeenDt = 0;
		vec2s numRxScanlines = { rxParameters->size(), (*rxParameters)[0].size() };
		m_numRxScanlines = numRxScanlines.x*numRxScanlines.y;
		m_rxScanlineLayout = numRxScanlines;
		size_t numElements = pTransducer->getNumElements();

		// create and fill new buffers
		m_pRxDepths = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, cudaStreamDefault, numDepths));

		m_pRxScanlines = unique_ptr<Container<ScanlineRxParameters3D> >(
			new Container<ScanlineRxParameters3D>(LocationHost, cudaStreamDefault, m_numRxScanlines));

		m_pRxElementXs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, cudaStreamDefault, numElements));
		m_pRxElementYs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, cudaStreamDefault, numElements));

		for (size_t zIdx = 0; zIdx < numDepths; zIdx++)
		{
			m_pRxDepths->get()[zIdx] = static_cast<LocationType>(zIdx*depth / (numDepths - 1));
		}
		size_t scanlineIdx = 0;
		for (size_t yIdx = 0; yIdx < numRxScanlines.y; yIdx++)
		{
			for (size_t xIdx = 0; xIdx < numRxScanlines.x; xIdx++)
			{
				m_pRxScanlines->get()[scanlineIdx] = (*rxParameters)[xIdx][yIdx];
				scanlineIdx++;
			}
		}

		auto centers = pTransducer->getElementCenterPoints();
		for (size_t x_elemIdx = 0; x_elemIdx < numElements; x_elemIdx++)
		{
			m_pRxElementXs->get()[x_elemIdx] = static_cast<LocationType>(centers->at(x_elemIdx).x);
			m_pRxElementYs->get()[x_elemIdx] = static_cast<LocationType>(centers->at(x_elemIdx).y);
		}

		m_pRxDepths = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxDepths));
		m_pRxScanlines = unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationGpu, *m_pRxScanlines));
		m_pRxElementXs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxElementXs));
		m_pRxElementYs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxElementYs));

		m_is3D = (pTransducer->getElementLayout().x > 1 && pTransducer->getElementLayout().y > 1);
	}

	RxBeamformerCuda::RxBeamformerCuda(
		size_t numRxScanlines,
		vec2s rxScanlineLayout,
		double speedOfSoundMMperS,
		const std::vector<LocationType> & rxDepths,
		const std::vector<ScanlineRxParameters3D> & rxScanlines,
		const std::vector<LocationType> & rxElementXs,
		const std::vector<LocationType> & rxElementYs,
		size_t rxNumDepths)
		: m_windowFunction(nullptr)
	{
		m_lastSeenDt = 0;
		m_numRxScanlines = numRxScanlines;
		m_rxScanlineLayout = rxScanlineLayout;

		m_is3D = (rxScanlineLayout.x > 1 && rxScanlineLayout.y > 1);
		m_speedOfSoundMMperS = speedOfSoundMMperS;
		m_rxNumDepths = rxNumDepths;

		// create and fill new buffers
		m_pRxDepths = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, cudaStreamDefault, rxDepths));

		m_pRxScanlines = unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationGpu, cudaStreamDefault, rxScanlines));

		m_pRxElementXs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, cudaStreamDefault, rxElementXs));
		m_pRxElementYs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, cudaStreamDefault, rxElementYs));
	}

	RxBeamformerCuda::~RxBeamformerCuda()
	{
	}

	void RxBeamformerCuda::convertToDtSpace(double dt, size_t numTransducerElements) const
	{
		if (m_lastSeenDt != dt)
		{
			double factor = 1;
			double factorTime = 1;
			if (m_lastSeenDt == 0)
			{
				factor = 1 / (m_speedOfSoundMMperS * dt);
				factorTime = 1 / dt;
			}
			else {
				factor = (m_lastSeenDt / dt);
				factorTime = factor;
			}
			m_pRxScanlines = unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationHost, *m_pRxScanlines));
			for (size_t i = 0; i < m_numRxScanlines; i++)
			{
				ScanlineRxParameters3D p = m_pRxScanlines->get()[i];
				p.position = p.position*factor;
				for (size_t k = 0; k < std::extent<decltype(p.txWeights)>::value; k++)
				{
					p.txParameters[k].initialDelay *= factorTime;
				}
				p.maxElementDistance = p.maxElementDistance*factor;
				m_pRxScanlines->get()[i] = p;
			}
			m_pRxScanlines = unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationGpu, *m_pRxScanlines));

			m_pRxDepths = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxDepths));
			for (size_t i = 0; i < m_rxNumDepths; i++)
			{
				m_pRxDepths->get()[i] = static_cast<LocationType>(m_pRxDepths->get()[i] * factor);
			}
			m_pRxDepths = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxDepths));

			m_pRxElementXs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxElementXs));
			m_pRxElementYs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxElementYs));
			for (size_t i = 0; i < numTransducerElements; i++)
			{
				m_pRxElementXs->get()[i] = static_cast<LocationType>(m_pRxElementXs->get()[i] * factor);
				m_pRxElementYs->get()[i] = static_cast<LocationType>(m_pRxElementYs->get()[i] * factor);
			}
			m_pRxElementXs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxElementXs));
			m_pRxElementYs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxElementYs));
			
			m_lastSeenDt = dt;
		}
	}

	template <typename T>
	__device__ inline T computeAperture_D(T F, T z)
	{
		return z / (2 * F);
	}

	template <typename T>
	__device__ inline T computeDelayDTSPACE_D(T dirX, T dirY, T dirZ, T x_element, T x, T z)
	{
		return sqrt((x_element - (x + dirX*z))*
			(x_element - (x + dirX*z)) +
			(dirY*z)*(dirY*z)) + z;
	}

	template <typename T>
	__device__ inline T computeDelayDTSPACE3D_D(T dirX, T dirY, T dirZ, T x_element, T z_element, T x, T z, T d)
	{
		return sqrt(
			squ(x_element - (x + dirX*d)) +
			squ(z_element - (z + dirZ*d)) +
			squ(dirY*d)) + d;
	}

	// distance has to be normalized to [-1, 1] (inclusive)
	__device__ inline WindowFunctionGpu::ElementType
		computeWindow3D(const WindowFunctionGpu& windowFunction, const vec2f& distance)
	{
		return
			sqrt(windowFunction.get(distance.x)*
				windowFunction.get(distance.y));
	}


	// distance has to be normalized to [-1, 1] (inclusive)
	__device__ inline WindowFunctionGpu::ElementType
		computeWindow3DShared(const WindowFunctionGpu& windowFunction, const WindowFunctionGpu::ElementType * __restrict__ sharedData, const vec2f& distance)
	{
		return
			sqrt(windowFunction.getShared(sharedData, distance.x)*
				windowFunction.getShared(sharedData, distance.y));
	}

	template <bool interpolateRFlines, bool interpolateBetweenTransmits, unsigned int maxNumElements, unsigned int maxNumFunctionElements, bool testSignal, typename RFType, typename ResultType, typename LocationType>
	__global__
		void rxBeamformingDTSPACE3DKernel(
			uint32_t numTransducerElements,
			vec2T<uint32_t> elementLayout,
			uint32_t numReceivedChannels,
			uint32_t numTimesteps,
			const RFType* __restrict__ RF,
			uint32_t numTxScanlines,
			uint32_t numRxScanlines,
			const ScanlineRxParameters3D* __restrict__ scanlinesDT,
			uint32_t numDs,
			const LocationType* __restrict__ dsDT,
			const LocationType* __restrict__ x_elemsDT,
			const LocationType* __restrict__ z_elemsDT,
			LocationType speedOfSound,
			LocationType dt,
			LocationType F,
			const WindowFunctionGpu windowFunction,
			ResultType* __restrict__ s)
	{
		__shared__ LocationType x_elemsDTsh[maxNumElements];
		__shared__ LocationType z_elemsDTsh[maxNumElements];
		__shared__ WindowFunction::ElementType functionShared[maxNumFunctionElements];
		//fetch element positions to shared memory
		for (int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;  //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
			threadId < maxNumElements && threadId < numTransducerElements;
			threadId += blockDim.x*blockDim.y)  //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		{
			x_elemsDTsh[threadId] = x_elemsDT[threadId];
			z_elemsDTsh[threadId] = z_elemsDT[threadId];
		}
		for (int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;  //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
			threadId < maxNumFunctionElements && threadId < windowFunction.numElements();
			threadId += blockDim.x*blockDim.y)  //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		{
			functionShared[threadId] = windowFunction.getDirect(threadId);
		}
		__syncthreads(); //@suppress("Function cannot be resolved")

		int r = blockDim.y * blockIdx.y + threadIdx.y; //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		int scanlineIdx = blockDim.x * blockIdx.x + threadIdx.x; //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")

		if (r < numDs && scanlineIdx < numRxScanlines)
		{
			LocationType d = dsDT[r];
			//TODO should this also depend on the angle?
			LocationType aDT = squ(computeAperture_D(F, d*dt*speedOfSound) / speedOfSound / dt);
			ScanlineRxParameters3D scanline = scanlinesDT[scanlineIdx];

			LocationType scanline_x = scanline.position.x;
			LocationType scanline_z = scanline.position.z;
			LocationType dirX = scanline.direction.x;
			LocationType dirY = scanline.direction.y;
			LocationType dirZ = scanline.direction.z;
			vec2f maxElementDistance = static_cast<vec2f>(scanline.maxElementDistance);
			vec2f invMaxElementDistance = vec2f{ 1.0f, 1.0f } / min(vec2f{ sqrt(aDT), sqrt(aDT) }, maxElementDistance);

			float sInterp = 0.0f;

			int highestWeightIndex;
			if (!interpolateBetweenTransmits)
			{
				highestWeightIndex = 0;
				float highestWeight = scanline.txWeights[0];
				for (int k = 1; k < std::extent<decltype(scanline.txWeights)>::value; k++)
				{
					if (scanline.txWeights[k] > highestWeight)
					{
						highestWeight = scanline.txWeights[k];
						highestWeightIndex = k;
					}
				}
			}

			// now iterate over all four txScanlines to interpolate beamformed scanlines from those transmits
			for (int k = (interpolateBetweenTransmits ? 0 : highestWeightIndex);
				(interpolateBetweenTransmits && k < std::extent<decltype(scanline.txWeights)>::value) ||
				(!interpolateBetweenTransmits && k == highestWeightIndex);
				k++)
			{
				if (scanline.txWeights[k] > 0.0)
				{
					ScanlineRxParameters3D::TransmitParameters txParams = scanline.txParameters[k];
					uint32_t txScanlineIdx = txParams.txScanlineIdx;
					if (txScanlineIdx >= numTxScanlines)
					{
						//ERROR!
						return;
					}
					LocationType initialDelay = static_cast<LocationType>(txParams.initialDelay);
					float sLocal = 0.0f;
					float weightAcum = 0.0f;
					int numAdds = 0;

					/*if (r == 0)
					{
						printf("#%d: tx scanline: %d, scanline x: %f, z: %f, dx: %f, dy: %f, dz: %f\n    firstIndX %d, firstIndY %d, lastIndX %d, lastIndY %d\n",
							scanlineIdx, txScanlineIdx, scanline_x, scanline_z, dirX, dirY, dirZ, scanline.firstActiveElementIndex.x, scanline.firstActiveElementIndex.y,
							scanline.lastActiveElementIndex.x, scanline.lastActiveElementIndex.y);

					}*/
					if (testSignal)
					{
						constexpr float cylinderSpacing = 6; //[mm]
						constexpr float cylinderDiameter = 2; //[mm]
						constexpr float cylinderDepth = 30; //[mm]
						constexpr int numCylindersHalf = 3;

						vec3f point = static_cast<vec3f>(scanline.position) + d * static_cast<vec3f>(scanline.direction);
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
								sLocal = 1000;
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
								sLocal = 1000;
							}
						}
					}
					else
					{
						for (uint32_t elemIdxX = txParams.firstActiveElementIndex.x; elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++)
						{
							for (uint32_t elemIdxY = txParams.firstActiveElementIndex.y; elemIdxY < txParams.lastActiveElementIndex.y; elemIdxY++)
							{
								uint32_t elemIdx = elemIdxX + elemIdxY*elementLayout.x;
								uint32_t  channelIdx = elemIdx % numReceivedChannels;
								LocationType x_elem = x_elemsDTsh[elemIdx];
								LocationType z_elem = z_elemsDTsh[elemIdx];

								if ((squ(x_elem - scanline_x) + squ(z_elem - scanline_z)) <= aDT)
								{
									vec2f elementScanlineDistance = { x_elem - scanline_x, z_elem - scanline_z };
									float weight = computeWindow3DShared(windowFunction, functionShared, elementScanlineDistance * invMaxElementDistance);
									weightAcum += weight;
									numAdds++;
									if (interpolateRFlines)
									{
										LocationType delayf = initialDelay +
											computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x, scanline_z, d);
										uint32_t delay = static_cast<uint32_t>(::floor(delayf));
										delayf -= delay;
										if (delay < (numTimesteps - 1))
										{
											sLocal +=
												weight * ((1.0f - delayf) * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps] +
													delayf  * RF[(delay + 1) + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps]);
										}
										else if (delay < numTimesteps && delayf == 0.0)
										{
											sLocal += weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
										}
									}
									else
									{
										uint32_t delay = static_cast<uint32_t>(::round(
											initialDelay + computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x, scanline_z, d)));
										if (delay < numTimesteps)
										{
											sLocal += weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
										}
									}
								}
							}
						}
					}
					if (interpolateBetweenTransmits)
					{
						sInterp += static_cast<float>(scanline.txWeights[k])* sLocal / weightAcum * numAdds;
					}
					else
					{
						sInterp += sLocal / weightAcum * numAdds;
					}
				}
			}
			s[scanlineIdx + r * numRxScanlines] = static_cast<ResultType>(sInterp);
		}
	}

	template <bool interpolateRFlines, bool interpolateBetweenTransmits, typename RFType, typename ResultType, typename LocationType>
	__global__
		void rxBeamformingDTSPACEKernel(
			size_t numTransducerElements,
			size_t numReceivedChannels,
			size_t numTimesteps,
			const RFType* __restrict__ RF,
			size_t numTxScanlines,
			size_t numRxScanlines,
			const ScanlineRxParameters3D* __restrict__ scanlinesDT,
			size_t numDs,
			const LocationType* __restrict__ dsDT,
			const LocationType* __restrict__ x_elemsDT,
			LocationType speedOfSound,
			LocationType dt,
			LocationType F,
			const WindowFunctionGpu windowFunction,
			ResultType* __restrict__ s)
	{
		int r = blockDim.y * blockIdx.y + threadIdx.y; //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		int scanlineIdx = blockDim.x * blockIdx.x + threadIdx.x; //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		if (r < numDs && scanlineIdx < numRxScanlines)
		{
			LocationType d = dsDT[r];
			//TODO should this also depend on the angle?
			LocationType aDT = computeAperture_D(F, d*dt*speedOfSound) / speedOfSound / dt;
			ScanlineRxParameters3D scanline = scanlinesDT[scanlineIdx];
			LocationType scanline_x = scanline.position.x;
			LocationType dirX = scanline.direction.x;
			LocationType dirY = scanline.direction.y;
			LocationType dirZ = scanline.direction.z;
			LocationType maxElementDistance = static_cast<LocationType>(scanline.maxElementDistance.x);
			LocationType invMaxElementDistance = 1 / min(aDT, maxElementDistance);

			float sInterp = 0.0f;

			int highestWeightIndex;
			if (!interpolateBetweenTransmits)
			{
				highestWeightIndex = 0;
				float highestWeight = scanline.txWeights[0];
				for (int k = 1; k < std::extent<decltype(scanline.txWeights)>::value; k++)
				{
					if (scanline.txWeights[k] > highestWeight)
					{
						highestWeight = scanline.txWeights[k];
						highestWeightIndex = k;
					}
				}
			}

			// now iterate over all four txScanlines to interpolate beamformed scanlines from those transmits
			for (int k = (interpolateBetweenTransmits ? 0 : highestWeightIndex);
				(interpolateBetweenTransmits && k < std::extent<decltype(scanline.txWeights)>::value) ||
				(!interpolateBetweenTransmits && k == highestWeightIndex);
				k++)
			{
				if (scanline.txWeights[k] > 0.0)
				{
					ScanlineRxParameters3D::TransmitParameters txParams = scanline.txParameters[k];
					uint32_t txScanlineIdx = txParams.txScanlineIdx;
					if (txScanlineIdx >= numTxScanlines)
					{
						//ERROR!
						return;
					}
					LocationType initialDelay = static_cast<LocationType>(txParams.initialDelay);

					float sLocal = 0.0f;
					float weightAcum = 0.0f;
					int numAdds = 0;
					for (int32_t elemIdxX = txParams.firstActiveElementIndex.x; elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++)
					{
						int32_t  channelIdx = elemIdxX % numReceivedChannels;
						LocationType x_elem = x_elemsDT[elemIdxX];
						if (abs(x_elem - scanline_x) <= aDT)
						{
							float weight = windowFunction.get((x_elem - scanline_x) * invMaxElementDistance);
							weightAcum += weight;
							numAdds++;
							if (interpolateRFlines)
							{
								LocationType delayf = initialDelay +
									computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, d);
								int32_t delay = static_cast<int32_t>(floor(delayf));
								delayf -= delay;
								if (delay < (numTimesteps - 1))
								{
									sLocal +=
										weight * ((1.0f - delayf) * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps] +
											delayf  * RF[(delay + 1) + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps]);
								}
								else if (delay < numTimesteps && delayf == 0.0)
								{
									sLocal += weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
								}
							}
							else
							{
								int32_t delay = static_cast<int32_t>(round(
									initialDelay + computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, d)));
								if (delay < numTimesteps)
								{
									sLocal += weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
								}
							}
						}
					}
					if (interpolateBetweenTransmits)
					{
						sInterp += static_cast<float>(scanline.txWeights[k])* sLocal / weightAcum * numAdds;
					}
					else
					{
						sInterp += sLocal / weightAcum * numAdds;
					}
				}
			}
			s[scanlineIdx + r * numRxScanlines] = static_cast<ResultType>(sInterp);
		}
	}

	template <unsigned int maxWindowFunctionNumel, typename RFType, typename ResultType, typename LocationType>
	void rxBeamformingDTspaceCuda3D(
		bool interpolateRFlines,
		bool interpolateBetweenTransmits,
		bool testSignal,
		size_t numTransducerElements,
		vec2s elementLayout,
		size_t numReceivedChannels,
		size_t numTimesteps,
		const RFType* RF,
		size_t numTxScanlines,
		size_t numRxScanlines,
		const ScanlineRxParameters3D* scanlines,
		size_t numZs,
		const LocationType* zs,
		const LocationType* x_elems,
		const LocationType* y_elems,
		LocationType speedOfSound,
		LocationType dt,
		LocationType F,
		const WindowFunctionGpu windowFunction,
		cudaStream_t stream,
		ResultType* s)
	{
		dim3 blockSize(1, 256);
		dim3 gridSize(
			static_cast<unsigned int>((numRxScanlines + blockSize.x - 1) / blockSize.x),
			static_cast<unsigned int>((numZs + blockSize.y - 1) / blockSize.y));
		if (testSignal)
		{
			rxBeamformingDTSPACE3DKernel<false, false, 1024, maxWindowFunctionNumel, true> << <gridSize, blockSize, 0, stream>> > (
				(uint32_t)numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout),
				(uint32_t)numReceivedChannels, (uint32_t)numTimesteps, RF,
				(uint32_t)numTxScanlines, (uint32_t)numRxScanlines, scanlines,
				(uint32_t)numZs, zs, x_elems, y_elems, speedOfSound, dt, F, windowFunction, s);
		}
		else
		{
			if (interpolateRFlines)
			{
				if (interpolateBetweenTransmits)
				{
					rxBeamformingDTSPACE3DKernel<true, true, 1024, maxWindowFunctionNumel, false> << <gridSize, blockSize, 0, stream>> > (
						(uint32_t)numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout),
						(uint32_t)numReceivedChannels, (uint32_t)numTimesteps, RF,
						(uint32_t)numTxScanlines, (uint32_t)numRxScanlines, scanlines,
						(uint32_t)numZs, zs, x_elems, y_elems, speedOfSound, dt, F, windowFunction, s);
				}
				else {
					rxBeamformingDTSPACE3DKernel<true, false, 1024, maxWindowFunctionNumel, false> << <gridSize, blockSize, 0, stream>> > (
						(uint32_t)numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout),
						(uint32_t)numReceivedChannels, (uint32_t)numTimesteps, RF,
						(uint32_t)numTxScanlines, (uint32_t)numRxScanlines, scanlines,
						(uint32_t)numZs, zs, x_elems, y_elems, speedOfSound, dt, F, windowFunction, s);
				}
			}
			else {
				if (interpolateBetweenTransmits)
				{
					rxBeamformingDTSPACE3DKernel<false, true, 1024, maxWindowFunctionNumel, false> << <gridSize, blockSize, 0, stream>> > (
						(uint32_t)numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout),
						(uint32_t)numReceivedChannels, (uint32_t)numTimesteps, RF,
						(uint32_t)numTxScanlines, (uint32_t)numRxScanlines, scanlines,
						(uint32_t)numZs, zs, x_elems, y_elems, speedOfSound, dt, F, windowFunction, s);
				}
				else {
					rxBeamformingDTSPACE3DKernel<false, false, 1024, maxWindowFunctionNumel, false> << <gridSize, blockSize, 0, stream>> > (
						(uint32_t)numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout),
						(uint32_t)numReceivedChannels, (uint32_t)numTimesteps, RF,
						(uint32_t)numTxScanlines, (uint32_t)numRxScanlines, scanlines,
						(uint32_t)numZs, zs, x_elems, y_elems, speedOfSound, dt, F, windowFunction, s);
				}
			}
		}
		cudaSafeCall(cudaPeekAtLastError());
	}

	template <typename RFType, typename ResultType, typename LocationType>
	void rxBeamformingDTspaceCuda(
		bool interpolateRFlines,
		bool interpolateBetweenTransmits,
		size_t numTransducerElements,
		size_t numReceivedChannels,
		size_t numTimesteps,
		const RFType* RF,
		size_t numTxScanlines,
		size_t numRxScanlines,
		const ScanlineRxParameters3D* scanlines,
		size_t numZs,
		const LocationType* zs,
		const LocationType* x_elems,
		LocationType speedOfSound,
		LocationType dt,
		LocationType F,
		const WindowFunctionGpu windowFunction,
		cudaStream_t stream,
		ResultType* s)
	{
		dim3 blockSize(1, 256);
		dim3 gridSize(
			static_cast<unsigned int>((numRxScanlines + blockSize.x - 1) / blockSize.x),
			static_cast<unsigned int>((numZs + blockSize.y - 1) / blockSize.y));
		if (interpolateRFlines)
		{
			if (interpolateBetweenTransmits)
			{
				rxBeamformingDTSPACEKernel<true, true> << <gridSize, blockSize, 0, stream>> > (
					numTransducerElements, numReceivedChannels, numTimesteps, RF,
					numTxScanlines, numRxScanlines, scanlines,
					numZs, zs, x_elems, speedOfSound, dt, F, windowFunction, s);
			}
			else {
				rxBeamformingDTSPACEKernel<true, false> << <gridSize, blockSize, 0, stream>> > (
					numTransducerElements, numReceivedChannels, numTimesteps, RF,
					numTxScanlines, numRxScanlines, scanlines,
					numZs, zs, x_elems, speedOfSound, dt, F, windowFunction, s);
			}
		}
		else {
			if (interpolateBetweenTransmits)
			{
				rxBeamformingDTSPACEKernel<false, true> << <gridSize, blockSize, 0, stream>> > (
					numTransducerElements, numReceivedChannels, numTimesteps, RF,
					numTxScanlines, numRxScanlines, scanlines,
					numZs, zs, x_elems, speedOfSound, dt, F, windowFunction, s);
			}
			else {
				rxBeamformingDTSPACEKernel<false, false> << <gridSize, blockSize, 0, stream>> > (
					numTransducerElements, numReceivedChannels, numTimesteps, RF,
					numTxScanlines, numRxScanlines, scanlines,
					numZs, zs, x_elems, speedOfSound, dt, F, windowFunction, s);
			}
		}
		cudaSafeCall(cudaPeekAtLastError());
	}

	template <>
	shared_ptr<USImage<int16_t> > RxBeamformerCuda::performRxBeamforming(
		shared_ptr<const USRawData<int16_t> > rawData,
		double fNumber,
		WindowType windowType,
		WindowFunction::ElementType windowParameter,
		bool interpolateBetweenTransmits,
		bool testSignal) const
	{
		//Ensure the raw-data are on the gpu
		auto gRawData = rawData->getData();
		if (!rawData->getData()->isGPU() && !rawData->getData()->isBoth())
		{
			gRawData = make_shared<Container<int16_t> >(LocationGpu, *gRawData);
		}

		size_t numelOut = m_numRxScanlines*m_rxNumDepths;
		shared_ptr<Container<int16_t> > pData = make_shared<Container<int16_t> >(ContainerLocation::LocationGpu, gRawData->getStream(), numelOut);

		double dt = 1.0 / rawData->getSamplingFrequency();

		if (!m_windowFunction || m_windowFunction->getType() != windowType || m_windowFunction->getParameter() != windowParameter)
		{
			m_windowFunction = std::unique_ptr<WindowFunction>(new WindowFunction(windowType, windowParameter, m_windowFunctionNumEntries));
		}

		convertToDtSpace(dt, rawData->getNumElements());
		if (m_is3D)
		{
			rxBeamformingDTspaceCuda3D<m_windowFunctionNumEntries, int16_t, int16_t, LocationType>(
				true,
				interpolateBetweenTransmits,
				testSignal,
				rawData->getNumElements(),
				rawData->getElementLayout(),
				rawData->getNumReceivedChannels(),
				rawData->getNumSamples(),
				gRawData->get(),
				rawData->getNumScanlines(), // numTxScanlines
				m_numRxScanlines,			// numRxScanlines
				m_pRxScanlines->get(),
				m_rxNumDepths, m_pRxDepths->get(),
				m_pRxElementXs->get(),
				m_pRxElementYs->get(),
				static_cast<LocationType>(m_speedOfSoundMMperS),
				static_cast<LocationType>(dt),
				static_cast<LocationType>(fNumber),
				*(m_windowFunction->getGpu()),
				gRawData->getStream(),
				pData->get()
				);
		}
		else {
			rxBeamformingDTspaceCuda<int16_t, int16_t, LocationType>(
				true,
				interpolateBetweenTransmits,
				rawData->getNumElements(),
				rawData->getNumReceivedChannels(),
				rawData->getNumSamples(),
				gRawData->get(),
				rawData->getNumScanlines(), // numTxScanlines
				m_numRxScanlines,			// numRxScanlines
				m_pRxScanlines->get(),
				m_rxNumDepths, m_pRxDepths->get(),
				m_pRxElementXs->get(),
				static_cast<LocationType>(m_speedOfSoundMMperS),
				static_cast<LocationType>(dt),
				static_cast<LocationType>(fNumber),
				*(m_windowFunction->getGpu()),
				gRawData->getStream(),
				pData->get()
				);
		}

		if (rawData->getImageProperties() != m_lastSeenImageProperties)
		{
			m_lastSeenImageProperties = rawData->getImageProperties();
			shared_ptr<USImageProperties> newProps = make_shared<USImageProperties>(*m_lastSeenImageProperties);
			newProps->setScanlineLayout(m_rxScanlineLayout);
			newProps->setNumSamples(m_rxNumDepths);
			newProps->setImageState(USImageProperties::RF);
			m_editedImageProperties = const_pointer_cast<const USImageProperties>(newProps);
		}

		auto retImage = make_shared<USImage<int16_t> >(
			vec2s{ m_numRxScanlines, m_rxNumDepths },
			pData,
			m_editedImageProperties,
			rawData->getReceiveTimestamp(),
			rawData->getSyncTimestamp());

		return retImage;
	}

	using std::setw;
	using std::setprecision;

	template <>
	void RxBeamformerCuda::writeMetaDataForMock(string filename, shared_ptr<const USRawData<int16_t> > rawData) const
	{
		std::ofstream f(filename);
		f << "rawDataMockMetadata v 3" << std::endl;
		f << rawData->getNumElements() << " "
			<< rawData->getElementLayout().x << " "
			<< rawData->getElementLayout().y << " "
			<< rawData->getNumReceivedChannels() << " "
			<< rawData->getNumSamples() << " "
			<< rawData->getNumScanlines() << " "
			<< m_rxScanlineLayout.x << " "
			<< m_rxScanlineLayout.y << " "
			<< rawData->getImageProperties()->getDepth() << " "
			<< rawData->getSamplingFrequency() << " "
			<< m_rxNumDepths << " "
			<< m_speedOfSoundMMperS << std::endl;

		auto rxScanlines = unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationHost, *m_pRxScanlines));
		auto rxDepths = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxDepths));
		auto rxElementXs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxElementXs));
		auto rxElementYs = unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxElementYs));

		for (size_t idx = 0; idx < m_numRxScanlines; idx++)
		{
			f << rxScanlines->get()[idx] << " ";
		}
		f << std::endl;
		for (size_t idx = 0; idx < m_rxNumDepths; idx++)
		{
			f << setprecision(9) << rxDepths->get()[idx] << " ";
		}
		f << std::endl;
		for (size_t idx = 0; idx < rawData->getNumElements(); idx++)
		{
			f << setprecision(9) << rxElementXs->get()[idx] << " ";
		}
		for (size_t idx = 0; idx < rawData->getNumElements(); idx++)
		{
			f << setprecision(9) << rxElementYs->get()[idx] << " ";
		}
		f << std::endl;
		f.close();
	}

	template<>
	shared_ptr<USRawData<int16_t> > RxBeamformerCuda::readMetaDataForMock(const std::string & mockMetadataFilename)
	{
		std::ifstream f(mockMetadataFilename);

		shared_ptr<USRawData<int16_t> > rawData;

		size_t numElements;
		size_t numReceivedChannels;
		size_t numSamples;
		size_t numTxScanlines;
		vec2s scanlineLayout;
		vec2s elementLayout;
		double depth;
		double samplingFrequency;
		size_t rxNumDepths;
		double speedOfSoundMMperS;
		//f << "rawDataMockMetadata v 1";
		string dummy;
		int version;
		f >> dummy;
		f >> dummy;
		f >> version;

		f >> numElements;
		f >> elementLayout.x;
		f >> elementLayout.y;
		f >> numReceivedChannels;
		f >> numSamples;
		f >> numTxScanlines;
		f >> scanlineLayout.x;
		f >> scanlineLayout.y;
		f >> depth;
		f >> samplingFrequency;
		f >> rxNumDepths;
		f >> speedOfSoundMMperS;

		size_t numRxScanlines = scanlineLayout.x*scanlineLayout.y;

		vector<ScanlineRxParameters3D> rxScanlines(numRxScanlines);
		vector<LocationType> rxDepths(rxNumDepths);
		vector<LocationType> rxElementXs(numElements);
		vector<LocationType> rxElementYs(numElements);

		shared_ptr<vector<vector<ScanlineRxParameters3D> > > scanlines =
			make_shared<vector<vector<ScanlineRxParameters3D> > >(scanlineLayout.x, vector<ScanlineRxParameters3D>(scanlineLayout.y));

		size_t scanlineIdx = 0;
		for (size_t idxY = 0; idxY < scanlineLayout.y; idxY++)
		{
			for (size_t idxX = 0; idxX < scanlineLayout.x; idxX++)
			{
				ScanlineRxParameters3D params;
				f >> params;
				(*scanlines)[idxX][idxY] = params;
				rxScanlines[scanlineIdx] = params;
				scanlineIdx++;
			}
		}

		for (size_t idx = 0; idx < rxNumDepths; idx++)
		{
			LocationType val;
			f >> val;
			rxDepths[idx] = val;
		}
		for (size_t idx = 0; idx < numElements; idx++)
		{
			LocationType val;
			f >> val;
			rxElementXs[idx] = val;
		}
		for (size_t idx = 0; idx < numElements; idx++)
		{
			LocationType val;
			f >> val;
			rxElementYs[idx] = val;
		}

		f.close();

		auto imageProps = make_shared<USImageProperties>(
			vec2s{ numTxScanlines, 1 },
			rxNumDepths,
			USImageProperties::ImageType::BMode,
			USImageProperties::ImageState::Raw,
			USImageProperties::TransducerType::Linear,
			depth);

		imageProps->setScanlineInfo(scanlines);

		auto rxBeamformer = shared_ptr<RxBeamformerCuda>(new RxBeamformerCuda(
			numRxScanlines,
			scanlineLayout,
			speedOfSoundMMperS,
			rxDepths,
			rxScanlines,
			rxElementXs,
			rxElementYs,
			rxNumDepths));

		auto pRawData = make_shared<USRawData<int16_t> >(
			numTxScanlines,
			numElements,
			elementLayout,
			numReceivedChannels,
			numSamples,
			samplingFrequency,
			nullptr,
			rxBeamformer,
			imageProps,
			0,
			0);

		return pRawData;
	}
}