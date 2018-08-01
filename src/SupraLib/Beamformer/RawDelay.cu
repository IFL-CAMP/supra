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
#include "RawDelay.h"
#include "USImage.h"
#include "USRawData.h"
#include "RxBeamformerCommon.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra
{
	RawDelay::RawDelay(const RxBeamformerParameters & parameters)
		: m_windowFunction(nullptr)
	{
		m_lastSeenDt = 0;
		m_numRxScanlines = parameters.getNumRxScanlines();
		m_rxScanlineLayout = parameters.getRxScanlineLayout();

		m_is3D = (m_rxScanlineLayout.x > 1 && m_rxScanlineLayout.y > 1);
		m_speedOfSoundMMperS = parameters.getSpeedOfSoundMMperS();
		m_rxNumDepths = parameters.getRxNumDepths();

		// create and fill new buffers
		auto depths = parameters.getRxDepths();
		m_depth = depths[depths.size() - 1];
		m_pRxDepths = std::unique_ptr<Container<LocationType> >(
			new Container<LocationType>(LocationGpu, cudaStreamDefault, depths));

		m_pRxScanlines = std::unique_ptr<Container<ScanlineRxParameters3D> >(
			new Container<ScanlineRxParameters3D>(LocationGpu, cudaStreamDefault, parameters.getRxScanlines()));

		m_pRxElementXs = std::unique_ptr<Container<LocationType> >(
			new Container<LocationType>(LocationGpu, cudaStreamDefault, parameters.getRxElementXs()));
		m_pRxElementYs = std::unique_ptr<Container<LocationType> >(
			new Container<LocationType>(LocationGpu, cudaStreamDefault, parameters.getRxElementYs()));
	}

	RawDelay::~RawDelay()
	{
	}

	void RawDelay::convertToDtSpace(double dt, double speedOfSoundMMperS, size_t numTransducerElements) const
	{
		if (m_lastSeenDt != dt || m_speedOfSoundMMperS != speedOfSoundMMperS)
		{
			double oldFactor = 1;
			double oldFactorTime = 1;
			if (m_lastSeenDt != 0 && m_speedOfSoundMMperS != 0)
			{
				oldFactor = 1 / (m_speedOfSoundMMperS * m_lastSeenDt);
				oldFactorTime = 1 / m_lastSeenDt;
			}

			double factor = 1/oldFactor / (speedOfSoundMMperS * dt);
			double factorTime = 1/oldFactorTime / dt;

			m_pRxScanlines = std::unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationHost, *m_pRxScanlines));
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
			m_pRxScanlines = std::unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationGpu, *m_pRxScanlines));

			m_pRxDepths = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxDepths));
			for (size_t i = 0; i < m_rxNumDepths; i++)
			{
				m_pRxDepths->get()[i] = static_cast<LocationType>(m_pRxDepths->get()[i] * factor);
			}
			m_pRxDepths = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxDepths));

			m_pRxElementXs = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxElementXs));
			m_pRxElementYs = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxElementYs));
			for (size_t i = 0; i < numTransducerElements; i++)
			{
				m_pRxElementXs->get()[i] = static_cast<LocationType>(m_pRxElementXs->get()[i] * factor);
				m_pRxElementYs->get()[i] = static_cast<LocationType>(m_pRxElementYs->get()[i] * factor);
			}
			m_pRxElementXs = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxElementXs));
			m_pRxElementYs = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxElementYs));
			
			m_lastSeenDt = dt;
			m_speedOfSoundMMperS = speedOfSoundMMperS;
		}
	}

	template <bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
	static __device__ void sampleDelay3D(
		ScanlineRxParameters3D::TransmitParameters txParams,
		int rxScanlineIdx,
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
		int depthIndex,
		uint32_t numTimestepsOut,
		vec2f invMaxElementDistance,
		LocationType speedOfSound,
		LocationType dt,
		const WindowFunctionGpu* windowFunction,
		const WindowFunction::ElementType* functionShared,
		ResultType* RFdelayed
	)
	{
		float sample = 0.0f;
		float weightAcum = 0.0f;
		int numAdds = 0;
		LocationType initialDelay = txParams.initialDelay;
		uint32_t txScanlineIdx = txParams.txScanlineIdx;

		// precompute the weighting factor
		for (uint32_t elemIdxX = txParams.firstActiveElementIndex.x; elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++)
		{
			for (uint32_t elemIdxY = txParams.firstActiveElementIndex.y; elemIdxY < txParams.lastActiveElementIndex.y; elemIdxY++)
			{
				uint32_t elemIdx = elemIdxX + elemIdxY*elementLayout.x;
				LocationType x_elem = x_elemsDTsh[elemIdx];
				LocationType z_elem = z_elemsDTsh[elemIdx];

				if ((squ(x_elem - scanline_x) + squ(z_elem - scanline_z)) <= aDT)
				{
					vec2f elementScanlineDistance = { x_elem - scanline_x, z_elem - scanline_z };
					float weight = computeWindow3DShared(*windowFunction, functionShared, elementScanlineDistance * invMaxElementDistance);
					weightAcum += weight;
					numAdds++;
				}
			}
		}
		float weightingScale = 1 / weightAcum * numAdds;

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
					float weight = computeWindow3DShared(*windowFunction, functionShared, elementScanlineDistance * invMaxElementDistance);
					if (interpolateRFlines)
					{
						LocationType delayf = initialDelay +
							computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x, scanline_z, depth);
						uint32_t delay = static_cast<uint32_t>(::floor(delayf));
						delayf -= delay;
						if (delay < (numTimesteps - 1))
						{
							sample =
								weight * ((1.0f - delayf) * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps] +
									delayf  * RF[(delay + 1) + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps]);
						}
						else if (delay < numTimesteps && delayf == 0.0)
						{
							sample = weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
						}
					}
					else
					{
						uint32_t delay = static_cast<uint32_t>(::round(
							initialDelay + computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x, scanline_z, depth)));
						if (delay < numTimesteps)
						{
							sample = weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
						}
					}
				}

				uint32_t elemIdxLocal = (elemIdxX - txParams.firstActiveElementIndex.x) + (elemIdxY - txParams.firstActiveElementIndex.y)*elementLayout.x;
				RFdelayed[depthIndex + elemIdxLocal*numTimestepsOut + rxScanlineIdx*numReceivedChannels*numTimestepsOut] =
					clampCast<ResultType>(sample * weightingScale);
			}
		}
	}

	template <bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
	static __device__ void sampleDelay2D(
		ScanlineRxParameters3D::TransmitParameters txParams,
		int rxScanlineIdx,
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
		int depthIndex,
		uint32_t numTimestepsOut,
		LocationType invMaxElementDistance,
		LocationType speedOfSound,
		LocationType dt,
		const WindowFunctionGpu* windowFunction,
		ResultType* RFdelayed
	)
	{
		float weightAcum = 0.0f;
		int numAdds = 0;
		LocationType initialDelay = txParams.initialDelay;
		uint32_t txScanlineIdx = txParams.txScanlineIdx;

		// precompute the accumulated weight
		for (int32_t elemIdxX = txParams.firstActiveElementIndex.x;
			elemIdxX < txParams.lastActiveElementIndex.x;
			elemIdxX++)
		{
			LocationType x_elem = x_elemsDT[elemIdxX];
			if (abs(x_elem - scanline_x) <= aDT)
			{
				float weight = windowFunction->get((x_elem - scanline_x) * invMaxElementDistance);
				weightAcum += weight;
				numAdds++;
			}
		}
		float weightingScale = 1 / weightAcum * numAdds;

		int32_t localElemIdxX = 0;
		for (int32_t elemIdxX = txParams.firstActiveElementIndex.x;
			 elemIdxX < txParams.lastActiveElementIndex.x;
			 elemIdxX++)
		{
			float sample = 0.0f;
			int32_t  channelIdx = elemIdxX % numReceivedChannels;
			LocationType x_elem = x_elemsDT[elemIdxX];
			if (abs(x_elem - scanline_x) <= aDT)
			{
				float weight = windowFunction->get((x_elem - scanline_x) * invMaxElementDistance);
				
				if (interpolateRFlines)
				{
					LocationType delayf = initialDelay +
						computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth);
					int32_t delay = static_cast<int32_t>(floor(delayf));
					delayf -= delay;
					if (delay < (numTimesteps - 1))
					{
						sample =
							weight * ((1.0f - delayf) * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps] +
								delayf  * RF[(delay + 1) + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps]);
					}
					else if (delay < numTimesteps && delayf == 0.0)
					{
						sample = weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
					}
				}
				else
				{
					int32_t delay = static_cast<int32_t>(round(
						initialDelay + computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth)));
					if (delay < numTimesteps)
					{
						sample = weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
					}
				}
			}
			RFdelayed[depthIndex + localElemIdxX*numTimestepsOut + rxScanlineIdx*numReceivedChannels*numTimestepsOut] =
				clampCast<ResultType>(sample * weightingScale);

			localElemIdxX++;
		}
	}
	
	template <bool interpolateRFlines, unsigned int maxNumElements, unsigned int maxNumFunctionElements, typename RFType, typename ResultType, typename LocationType>
	__global__
		void rxDelayDTSPACE3DKernel(
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
			ResultType* __restrict__ RFdelayed)
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

			int highestWeightIndex = 0;
			float highestWeight = scanline.txWeights[0];
			for (int k = 1; k < std::extent<decltype(scanline.txWeights)>::value; k++)
			{
				if (scanline.txWeights[k] > highestWeight)
				{
					highestWeight = scanline.txWeights[k];
					highestWeightIndex = k;
				}
			}

			{
				int k = highestWeightIndex;
				if (scanline.txWeights[k] > 0.0)
				{
					ScanlineRxParameters3D::TransmitParameters txParams = scanline.txParameters[k];
					uint32_t txScanlineIdx = txParams.txScanlineIdx;
					if (txScanlineIdx >= numTxScanlines)
					{
						//ERROR!
						return;
					}
					
					sampleDelay3D<interpolateRFlines, RFType, ResultType, LocationType>(
						txParams, scanlineIdx, RF, elementLayout, numReceivedChannels, numTimesteps,
						x_elemsDTsh, z_elemsDTsh, scanline_x, scanline_z, dirX, dirY, dirZ,
						aDT, d, r, numDs, invMaxElementDistance, speedOfSound, dt, &windowFunction, functionShared, RFdelayed);
				}
			}
		}
	}

	template <bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
	__global__
		void rxDelayDTSPACEKernel(
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
			ResultType* __restrict__ RFdelayed)
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

			int highestWeightIndex = 0;
			float highestWeight = scanline.txWeights[0];
			for (int k = 1; k < std::extent<decltype(scanline.txWeights)>::value; k++)
			{
				if (scanline.txWeights[k] > highestWeight)
				{
					highestWeight = scanline.txWeights[k];
					highestWeightIndex = k;
				}
			}

			{
				int k = highestWeightIndex;
				if (scanline.txWeights[k] > 0.0)
				{
					ScanlineRxParameters3D::TransmitParameters txParams = scanline.txParameters[k];
					uint32_t txScanlineIdx = txParams.txScanlineIdx;
					if (txScanlineIdx >= numTxScanlines)
					{
						//ERROR!
						return;
					}

					sampleDelay2D<interpolateRFlines, RFType, ResultType, LocationType>(
						txParams, scanlineIdx, RF, numTransducerElements, numReceivedChannels, numTimesteps,
						x_elemsDT, scanline_x, dirX, dirY, dirZ,
						aDT, d, r, numDs, invMaxElementDistance, speedOfSound, dt, &windowFunction, RFdelayed);
				}
			}
		}
	}

	template <unsigned int maxWindowFunctionNumel, typename RFType, typename OutputType, typename LocationType>
	void rxDelayDTspaceCuda3D(
		bool interpolateRFlines,
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
		OutputType* s)
	{
		dim3 blockSize(1, 256);
		dim3 gridSize(
			static_cast<unsigned int>((numRxScanlines + blockSize.x - 1) / blockSize.x),
			static_cast<unsigned int>((numZs + blockSize.y - 1) / blockSize.y));

		if (interpolateRFlines)
		{
			rxDelayDTSPACE3DKernel<true, 1024, maxWindowFunctionNumel> << <gridSize, blockSize, 0, stream>> > (
				(uint32_t)numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout),
				(uint32_t)numReceivedChannels, (uint32_t)numTimesteps, RF,
				(uint32_t)numTxScanlines, (uint32_t)numRxScanlines, scanlines,
				(uint32_t)numZs, zs, x_elems, y_elems, speedOfSound, dt, F, windowFunction, s);
		}
		else {
			rxDelayDTSPACE3DKernel<false, 1024, maxWindowFunctionNumel> << <gridSize, blockSize, 0, stream>> > (
				(uint32_t)numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout),
				(uint32_t)numReceivedChannels, (uint32_t)numTimesteps, RF,
				(uint32_t)numTxScanlines, (uint32_t)numRxScanlines, scanlines,
				(uint32_t)numZs, zs, x_elems, y_elems, speedOfSound, dt, F, windowFunction, s);
		}
		cudaSafeCall(cudaPeekAtLastError());
	}

	template <typename RFType, typename OutputType, typename LocationType>
	void rxDelayDTspaceCuda(
		bool interpolateRFlines,
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
		OutputType* s)
	{
		dim3 blockSize(1, 256);
		dim3 gridSize(
			static_cast<unsigned int>((numRxScanlines + blockSize.x - 1) / blockSize.x),
			static_cast<unsigned int>((numZs + blockSize.y - 1) / blockSize.y));
		if (interpolateRFlines)
		{
			rxDelayDTSPACEKernel<true> <<<gridSize, blockSize, 0, stream>>> (
				numTransducerElements, numReceivedChannels, numTimesteps, RF,
				numTxScanlines, numRxScanlines, scanlines,
				numZs, zs, x_elems, speedOfSound, dt, F, windowFunction, s);
		}
		else {
			rxDelayDTSPACEKernel<false> <<<gridSize, blockSize, 0, stream>>> (
				numTransducerElements, numReceivedChannels, numTimesteps, RF,
				numTxScanlines, numRxScanlines, scanlines,
				numZs, zs, x_elems, speedOfSound, dt, F, windowFunction, s);
		}
		cudaSafeCall(cudaPeekAtLastError());
	}

	template <typename ChannelDataType, typename OutputType>
	shared_ptr<USRawData> RawDelay::performDelay(
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameter) const
	{
		//Ensure the raw-data are on the gpu
		auto gRawData = rawData->getData<ChannelDataType>();
		if (!rawData->getData<ChannelDataType>()->isGPU() && !rawData->getData<ChannelDataType>()->isBoth())
		{
			gRawData = std::make_shared<Container<ChannelDataType> >(LocationGpu, *gRawData);
		}

		size_t numelOut = m_numRxScanlines*m_rxNumDepths*rawData->getNumReceivedChannels();
		shared_ptr<Container<OutputType> > pData = 
			std::make_shared<Container<OutputType> >(ContainerLocation::LocationGpu, gRawData->getStream(), numelOut);

		double dt = 1.0 / rawData->getSamplingFrequency();

		if (!m_windowFunction || m_windowFunction->getType() != windowType || m_windowFunction->getParameter() != windowParameter)
		{
			m_windowFunction = std::unique_ptr<WindowFunction>(new WindowFunction(windowType, windowParameter, m_windowFunctionNumEntries));
		}
		
		convertToDtSpace(dt, speedOfSoundMMperS, rawData->getNumElements());
		if (m_is3D)
		{
			rxDelayDTspaceCuda3D<m_windowFunctionNumEntries, ChannelDataType, OutputType, LocationType>(
				true,
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
			rxDelayDTspaceCuda<ChannelDataType, OutputType, LocationType>(
				true,
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
			shared_ptr<USImageProperties> newProps = std::make_shared<USImageProperties>(*m_lastSeenImageProperties);
			newProps->setScanlineLayout(m_rxScanlineLayout);
			newProps->setNumSamples(m_rxNumDepths);
			newProps->setImageState(USImageProperties::RF);
			m_editedImageProperties = std::const_pointer_cast<const USImageProperties>(newProps);
		}

		shared_ptr<USRawData> rawDataDelayed = std::make_shared<USRawData>
			   (m_numRxScanlines,
				rawData->getNumElements(),
				rawData->getElementLayout(),
				rawData->getNumReceivedChannels(),
				m_rxNumDepths,
				1 / (m_depth / m_speedOfSoundMMperS / m_rxNumDepths),
				pData,
				rawData->getRxBeamformerParameters(),
				m_editedImageProperties,
				rawData->getReceiveTimestamp(),
				rawData->getSyncTimestamp());

		return rawDataDelayed;
	}

	template
	shared_ptr<USRawData> RawDelay::performDelay<int16_t, int16_t>(
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameters) const;
	template
	shared_ptr<USRawData> RawDelay::performDelay<int16_t, float>(
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameters) const;
	template
	shared_ptr<USRawData> RawDelay::performDelay<float, int16_t>(
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameters) const;
	template
	shared_ptr<USRawData> RawDelay::performDelay<float, float>(
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameters) const;
}