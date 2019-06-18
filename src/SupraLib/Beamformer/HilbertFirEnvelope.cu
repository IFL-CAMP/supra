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

#include "HilbertFirEnvelope.h"
#include <utilities/utility.h>
#include <utilities/FirFilterFactory.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <algorithm>

using namespace std;
using namespace thrust::placeholders;

namespace supra
{
	template <typename InputType, typename OutputType>
	__global__ void kernelFilterDemodulation(
		const InputType* __restrict__ signal,
		const HilbertFirEnvelope::WorkType * __restrict__ filter,
		OutputType * __restrict__ out,
		const int numSamples,
		const int numScanlines,
		const int filterLength) {
		int scanlineIdx = blockDim.x*blockIdx.x + threadIdx.x;
		int sampleIdx = blockDim.y*blockIdx.y + threadIdx.y;

		if (scanlineIdx < numScanlines && sampleIdx < numSamples)
		{
			HilbertFirEnvelope::WorkType accumulator = 0;
			
			int startPoint = sampleIdx - filterLength / 2;
			int endPoint = sampleIdx + filterLength / 2;
			int currentFilterElement = 0;
			for (int currentSample = startPoint;
				currentSample <= endPoint;
				currentSample++, currentFilterElement++)
			{
				if (currentSample >= 0 && currentSample < numSamples)
				{
					HilbertFirEnvelope::WorkType sample = static_cast<HilbertFirEnvelope::WorkType>(signal[scanlineIdx + currentSample*numScanlines]);
					HilbertFirEnvelope::WorkType filterElement = filter[currentFilterElement];
					accumulator += sample*filterElement;
				}
			}

			HilbertFirEnvelope::WorkType signalValue = static_cast<HilbertFirEnvelope::WorkType>(signal[scanlineIdx + sampleIdx*numScanlines]);
			out[scanlineIdx + sampleIdx*numScanlines] = sqrt(squ(signalValue) + squ(accumulator));
		}

	}

	HilbertFirEnvelope::HilbertFirEnvelope(size_t filterLength)
		: m_filterLength(filterLength)
		, m_hilbertFilter(nullptr)
	{
		prepareFilter();
	}

	HilbertFirEnvelope::~HilbertFirEnvelope()
	{
	}

	void HilbertFirEnvelope::prepareFilter()
	{
		m_hilbertFilter = FirFilterFactory::createFilter<float>(
			m_filterLength,
			FirFilterFactory::FilterTypeHilbertTransformer,
			FirFilterFactory::FilterWindowHamming);
		m_hilbertFilter = make_shared<Container<float> >(LocationGpu, *m_hilbertFilter);
	}

	template<typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > HilbertFirEnvelope::demodulate(
		const shared_ptr<const Container<InputType>>& inImageData,
		int numScanlines, int numSamples)
	{
		auto pEnv = make_shared<Container<OutputType> >(LocationGpu, inImageData->getStream(), numScanlines*numSamples);
		dim3 blockSizeFilter(16, 8);
		dim3 gridSizeFilter(
			static_cast<unsigned int>((numScanlines + blockSizeFilter.x - 1) / blockSizeFilter.x),
			static_cast<unsigned int>((numSamples + blockSizeFilter.y - 1) / blockSizeFilter.y));

		kernelFilterDemodulation<<<gridSizeFilter, blockSizeFilter, 0, inImageData->getStream()>>> (
			inImageData->get(),
			m_hilbertFilter->get(),
			pEnv->get(),
			numSamples,
			numScanlines,
			(int)m_filterLength);
		cudaSafeCall(cudaPeekAtLastError());

		return pEnv;
	}

	template 
	shared_ptr<Container<int16_t> > HilbertFirEnvelope::demodulate<int16_t, int16_t>(
		const shared_ptr<const Container<int16_t> >& inImageData,
		int numScanlines, int numSamples);
	template
		shared_ptr<Container<int16_t> > HilbertFirEnvelope::demodulate<float, int16_t>(
			const shared_ptr<const Container<float> >& inImageData,
			int numScanlines, int numSamples);
	template
		shared_ptr<Container<float> > HilbertFirEnvelope::demodulate<int16_t, float>(
			const shared_ptr<const Container<int16_t> >& inImageData,
			int numScanlines, int numSamples);
	template
		shared_ptr<Container<float> > HilbertFirEnvelope::demodulate<float, float>(
			const shared_ptr<const Container<float> >& inImageData,
			int numScanlines, int numSamples);
}