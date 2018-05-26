// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2018, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "HilbertEnvelope.h"
#include <utilities/utility.h>
#include <utilities/FirFilterFactory.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <algorithm>
#include <complex>

using namespace std;
using namespace thrust::placeholders;

namespace supra
{
	namespace hilbertEnvelopeKernels
	{
		template <typename InputType>
		__global__ void maskSpectrum(
			InputType* __restrict__ in,
			const int numSamples,
			const int numScanlines)
		{
			int scanlineIdx = blockDim.x*blockIdx.x + threadIdx.x;
			int sampleIdx = blockDim.y*blockIdx.y + threadIdx.y;

			if (scanlineIdx < numScanlines && sampleIdx < numSamples)
			{
				InputType val = in[sampleIdx + scanlineIdx*numSamples];
				float factor = 1.0f;
				if (numSamples % 2 == 0)
				{
					if (sampleIdx == 0 || sampleIdx == numSamples / 2)
					{
						// multiply by one, duh!
						factor = 1.0f;
					}
					else if (sampleIdx < numSamples / 2)
					{
						// multiply by two
						factor = 2.0f;
					}
					else
					{
						factor = 0.0f;
					}

				}
				else {
					if (sampleIdx == 0)
					{
						// multiply by one, duh!
						factor = 1.0f;
					}
					else if (sampleIdx < (numSamples + 1) / 2)
					{
						factor = 2.0f;
					}
					else {
						factor = 0.0f;
					}
				}
				val = make_cuFloatComplex(factor*cuCrealf(val), factor*cuCimagf(val));
				in[sampleIdx + scanlineIdx*numSamples] = val;
			}
		}

		template <typename InputType, typename OutputType>
		__global__ void absOfComplex(
			const InputType* __restrict__ in,
			OutputType* __restrict__ out,
			const int numSamples,
			const int numScanlines)
		{
			int scanlineIdx = blockDim.x*blockIdx.x + threadIdx.x;
			int sampleIdx = blockDim.y*blockIdx.y + threadIdx.y;

			if (scanlineIdx < numScanlines && sampleIdx < numSamples)
			{
				//InputType val = in[scanlineIdx + sampleIdx*numScanlines];
				//TEST transposing
				InputType val = in[sampleIdx + scanlineIdx*numSamples];
				//cufftComplex valC = val;
				//float valAbs = sqrt(squ(valC.x) + squ(valC.y));
				float valAbs = cuCabsf(val) / numSamples;
				out[scanlineIdx + sampleIdx*numScanlines] = clampCast<OutputType>(valAbs);
			}
		}

		template <typename InputType, typename OutputType>
		__global__ void convertInput(
			const InputType* __restrict__ in,
			OutputType* __restrict__ out,
			const int numSamples,
			const int numScanlines)
		{
			int scanlineIdx = blockDim.x*blockIdx.x + threadIdx.x;
			int sampleIdx = blockDim.y*blockIdx.y + threadIdx.y;

			if (scanlineIdx < numScanlines && sampleIdx < numSamples)
			{
				InputType val = in[scanlineIdx + sampleIdx*numScanlines];
				//out[scanlineIdx + sampleIdx*numScanlines] = clampCast<OutputType>(val);
				// TEST transposing
				//out[sampleIdx + scanlineIdx*numSamples] = clampCast<OutputType>(val);
				out[sampleIdx + scanlineIdx*numSamples] = make_cuFloatComplex(val, 0.0f);
			}
		}
	}

	HilbertEnvelope::HilbertEnvelope()
		: m_fftPlanLength(0)
		, m_fftPlanBatch(0)
		, m_fftHavePlan(false)
	{
	}

	HilbertEnvelope::~HilbertEnvelope()
	{
		cufftSafeCall(cufftDestroy(m_cufftHandleR2C));
		cufftSafeCall(cufftDestroy(m_cufftHandleC2C));
	}

	int HilbertEnvelope::decimatedSignalLength(int numSamples, uint32_t decimation)
	{
		int samplesAfterDecimation = numSamples;
		if (decimation > 1)
		{
			logging::log_warn("HilbertEnvelope: decimation currently not supported");
			//samplesAfterDecimation = numSamples / decimation;
		}
		return samplesAfterDecimation;
	}

	template<typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > HilbertEnvelope::computeHilbertEnvelope(
		const shared_ptr<const Container<InputType>>& inImageData,
		int numScanlines, int numSamples, uint32_t decimation)
	{

		if (numScanlines != m_fftPlanBatch || numSamples != m_fftPlanLength)
		{
			m_fftPlanBatch = numScanlines;
			m_fftPlanLength = numSamples;

			if (m_fftHavePlan)
			{
				cufftSafeCall(cufftDestroy(m_cufftHandleR2C));
				cufftSafeCall(cufftDestroy(m_cufftHandleC2C));
			}

			cufftSafeCall(cufftPlan1d(&m_cufftHandleR2C, m_fftPlanLength, CUFFT_R2C, m_fftPlanBatch));
			cufftSafeCall(cufftPlan1d(&m_cufftHandleC2C, m_fftPlanLength, CUFFT_C2C, m_fftPlanBatch));
		}
		
		cufftSafeCall(cufftSetStream(m_cufftHandleR2C, inImageData->getStream()));
		cufftSafeCall(cufftSetStream(m_cufftHandleC2C, inImageData->getStream()));

		// converted input
		auto pInput = make_shared<Container<cufftComplex> >(LocationGpu, inImageData->getStream(), numScanlines*numSamples);
		// fft of input data
		auto pFft = make_shared<Container<cufftComplex> >(LocationGpu, inImageData->getStream(), numScanlines*numSamples);
		// analytic signal
		auto pAnalyticSignal = make_shared<Container<cufftComplex> >(LocationGpu, inImageData->getStream(), numScanlines*numSamples);
		// the output
		auto pEnv = make_shared<Container<OutputType> >(LocationGpu, inImageData->getStream(), numScanlines*numSamples);

		dim3 blockSizeFilter(16, 8);
		dim3 gridSizeFilter(
			static_cast<unsigned int>((numScanlines + blockSizeFilter.x - 1) / blockSizeFilter.x),
			static_cast<unsigned int>((numSamples + blockSizeFilter.y - 1) / blockSizeFilter.y));

		// convert input
		hilbertEnvelopeKernels::convertInput <<<gridSizeFilter, blockSizeFilter, 0, inImageData->getStream() >>> (
			inImageData->get(),
			pInput->get(),
			numSamples,
			numScanlines);
		cudaSafeCall(cudaPeekAtLastError());

		// compute fft
		//cufftSafeCall(cufftExecR2C(m_cufftHandleR2C, pInput->get(), pFft->get()));
		cufftSafeCall(cufftExecC2C(m_cufftHandleC2C, pInput->get(), pFft->get(), CUFFT_FORWARD));
		
		// mask spectrum
		hilbertEnvelopeKernels::maskSpectrum <<<gridSizeFilter, blockSizeFilter, 0, pFft->getStream()>>> (
			pFft->get(),
			numSamples,
			numScanlines);
		cudaSafeCall(cudaPeekAtLastError());

		// compute ifft
		cufftSafeCall(cufftExecC2C(m_cufftHandleC2C, pFft->get(), pAnalyticSignal->get(), CUFFT_INVERSE));

		// compute abs
		hilbertEnvelopeKernels::absOfComplex <<<gridSizeFilter, blockSizeFilter, 0, pAnalyticSignal->getStream() >>> (
			pAnalyticSignal->get(),
			pEnv->get(),
			numSamples,
			numScanlines);
		cudaSafeCall(cudaPeekAtLastError());

		return pEnv;
	}

	template 
	shared_ptr<Container<int16_t> > HilbertEnvelope::computeHilbertEnvelope<int16_t, int16_t>(
		const shared_ptr<const Container<int16_t> >& inImageData,
		int numScanlines, int numSamples, uint32_t decimation);
	template
	shared_ptr<Container<int16_t> > HilbertEnvelope::computeHilbertEnvelope<float, int16_t>(
		const shared_ptr<const Container<float> >& inImageData,
		int numScanlines, int numSamples, uint32_t decimation);
	template
	shared_ptr<Container<float> > HilbertEnvelope::computeHilbertEnvelope<int16_t, float>(
		const shared_ptr<const Container<int16_t> >& inImageData,
		int numScanlines, int numSamples, uint32_t decimation);
	template
	shared_ptr<Container<float> > HilbertEnvelope::computeHilbertEnvelope<float, float>(
		const shared_ptr<const Container<float> >& inImageData,
		int numScanlines, int numSamples, uint32_t decimation);
}