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

#include "IQDemodulator.h"
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
	template <typename In, typename Out>
	struct thrustSqrt
	{
		__host__ __device__ Out operator()(const In& a, const In& b)
		{
			return (Out)(sqrt(a*a + b*b));
		}
	};

	template <typename In, typename Out>
	struct thrustScale
	{
		In _scale;
		thrustScale(In scale)
			:_scale(scale) {};
		__host__ __device__ Out operator()(const In& a)
		{
			return (Out)(_scale*a);
		}
	};

	template <typename In, typename Out>
	struct thrustCast
	{
		__host__ __device__ Out operator()(const In& a)
		{
			return static_cast<Out>(a);
		}
	};

	// For computing the phase from IQ
	template <typename In, typename Out>
	struct thrustAtan2
	{
		__host__ __device__ Out operator()(const In& a, const In& b)
		{
			return (Out)(atan2(b, a));
		}
	};

	template <int maxNumFilters, uint32_t iPhaseNumerator, uint32_t iPhaseDenominator, uint32_t qPhaseNumerator, uint32_t qPhaseDenominator, typename InputType>
	__global__ void kernelFilterBankDemodulation(
		const InputType* __restrict__ signal,
		const IQDemodulator::WorkType * __restrict__ filters,
		IQDemodulator::WorkType * __restrict__ out,
		const int numSamples,
		const int numScanlines,
		const int filterLength,
		const int bankOffset,
		const int numFilters, 
		const IQDemodulator::WorkType * __restrict__ weights,
		const IQDemodulator::WorkType scale,
		const IQDemodulator::WorkType * __restrict__ referenceFrequencyOverSamplingFrequencies) {
		int scanlineIdx = blockDim.x*blockIdx.x + threadIdx.x;
		int sampleIdx = blockDim.y*blockIdx.y + threadIdx.y;

		if (scanlineIdx < numScanlines && sampleIdx < numSamples)
		{
			IQDemodulator::WorkType accumulators[maxNumFilters];
			// Set the target values to zero
			for (int filter = 0; filter < numFilters; filter++)
			{
				accumulators[filter] = 0;
			}

			float signalValue = static_cast<float>(signal[scanlineIdx + sampleIdx*numScanlines]);
			for (int filter = 0; filter < bankOffset; filter++)
			{
				accumulators[filter] = signalValue;
			}

			int startPoint = sampleIdx - filterLength / 2;
			int endPoint = sampleIdx + filterLength / 2;
			int currentFilterElement = 0;
			for (int currentSample = startPoint;
				currentSample <= endPoint;
				currentSample++, currentFilterElement++)
			{
				if (currentSample >= 0 && currentSample < numSamples)
				{
					IQDemodulator::WorkType sample = signal[scanlineIdx + currentSample*numScanlines];
					for (int filter = bankOffset; filter < numFilters; filter++)
					{
						IQDemodulator::WorkType filterElement = filters[currentFilterElement + filterLength*filter];
						accumulators[filter] += sample*filterElement;
					}
				}
			}

			IQDemodulator::WorkType accumulator = 0;
			for (int filter = 0; filter < numFilters; filter++)
			{
				IQDemodulator::WorkType signalValue = accumulators[filter];

				IQDemodulator::WorkType referenceSignalPhase = sampleIdx * referenceFrequencyOverSamplingFrequencies[filter];
				IQDemodulator::WorkType referenceSignalPhaseI = referenceSignalPhase + static_cast<IQDemodulator::WorkType>(iPhaseNumerator) / static_cast<IQDemodulator::WorkType>(iPhaseDenominator);
				IQDemodulator::WorkType referenceSignalPhaseQ = referenceSignalPhase + static_cast<IQDemodulator::WorkType>(qPhaseNumerator) / static_cast<IQDemodulator::WorkType>(qPhaseDenominator);
				referenceSignalPhaseI = referenceSignalPhaseI - truncf(referenceSignalPhaseI);
				referenceSignalPhaseQ = referenceSignalPhaseQ - truncf(referenceSignalPhaseQ);
				IQDemodulator::WorkType referenceSignalI = cos(referenceSignalPhaseI*2.0f*static_cast<IQDemodulator::WorkType>(M_PI));
				IQDemodulator::WorkType referenceSignalQ = cos(referenceSignalPhaseQ*2.0f*static_cast<IQDemodulator::WorkType>(M_PI));

				IQDemodulator::WorkType signalValueDownmixedI = signalValue * referenceSignalI;
				IQDemodulator::WorkType signalValueDownmixedQ = signalValue * referenceSignalQ;
				accumulator += weights[filter] * sqrt(squ(signalValueDownmixedI) + squ(signalValueDownmixedQ));
			}

			out[scanlineIdx + sampleIdx*numScanlines] = scale * accumulator;
		}

	}

	// Input: Signal x bankSize
	// Computes: weightedSum((in[k] * (complex) referenceFreqOverSamplingFreq[k]) x  filter[0] (at stride locations))
	template <typename InputType, typename OutputType>
	__global__ void kernelFilterStrided(
		const InputType* __restrict__ in,
		const IQDemodulator::WorkType * __restrict__ filter,
		OutputType * __restrict__ out,
		const int numSamples,
		const int numScanlines,
		const int filterLength,
		const uint32_t stride) {
		int scanlineIdx = blockDim.x*blockIdx.x + threadIdx.x;
		int sampleIdxOut = blockDim.y*blockIdx.y + threadIdx.y;
		int sampleIdxIn = sampleIdxOut*stride;

		if (scanlineIdx < numScanlines && sampleIdxIn < numSamples)
		{
			float accumulator = 0;

			int startPoint = sampleIdxIn - filterLength / 2;
			int endPoint = sampleIdxIn + filterLength / 2;
			int currentFilterElement = 0;
			for (int currentSample = startPoint;
				currentSample <= endPoint;
				currentSample++, currentFilterElement++)
			{
				if (currentSample >= 0 && currentSample < numSamples)
				{
					IQDemodulator::WorkType filterValue = filter[currentFilterElement];
					IQDemodulator::WorkType sample = in[scanlineIdx + currentSample*numScanlines];
					accumulator += sample * filterValue;
				}
			}

			out[scanlineIdx + sampleIdxOut*numScanlines] = clampCast<OutputType>(accumulator);
		}
	}



	IQDemodulator::IQDemodulator(double samplingFrequency, double referenceFrequency, double cutoffFrequency, size_t lowpassFilterLength, size_t bandpassFilterLength)
		: m_samplingFrequency(samplingFrequency)
		//, m_referenceFrequency(referenceFrequency)
		, m_cutoffFrequency(cutoffFrequency)
		, m_decimationLowpassFilterLength(lowpassFilterLength)
		, m_frequencyCompoundingBandpassFilterLength(bandpassFilterLength)
		, m_frequencyCompoundingBandpassFilterBank(nullptr)
	{
		prepareFilter();
	}

	IQDemodulator::~IQDemodulator()
	{
	}

	void IQDemodulator::prepareFilter()
	{
		m_decimationLowpassFilter = FirFilterFactory::createFilter<float>(
			m_decimationLowpassFilterLength,
			FirFilterFactory::FilterTypeLowPass,
			FirFilterFactory::FilterWindowHamming,
			m_samplingFrequency, m_cutoffFrequency);
		m_decimationLowpassFilter = make_shared<Container<float> >(LocationGpu, *m_decimationLowpassFilter);
	}

	int IQDemodulator::decimatedSignalLength(int numSamples, uint32_t decimation)
	{
		int samplesAfterDecimation = numSamples;
		if (decimation > 1)
		{
			samplesAfterDecimation = numSamples / decimation;
		}
		return samplesAfterDecimation;
	}

	template<typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > IQDemodulator::demodulateMagnitudeFrequencyCompounding(
		const shared_ptr<const Container<InputType>>& inImageData,
		int numScanlines, int numSamples, uint32_t decimation,
		const std::vector<double>& referenceFrequencies,
		const std::vector<double>& bandwidths,
		const std::vector<double>& weights)
	{
		constexpr int maxNumbandpassFilters = 8;
		assert(referenceFrequencies.size() <= maxNumbandpassFilters);
		assert(referenceFrequencies.size() == bandwidths.size());
		assert(referenceFrequencies.size() == weights.size());

		WorkType weightAcum = 0.0;

		// check that the bandpass filters are still up to date
		bool bankNeedsUpdate = false;
		bool weightsNeedUpdate = false;
		m_frequencyCompoundingReferenceFrequencies.resize(referenceFrequencies.size(), 0);
		m_frequencyCompoundingBandwidths.resize(bandwidths.size(), 0);
		m_frequencyCompoundingWeights.resize(weights.size(), 0);
		m_frequencyCompoundingBandpassFilters.resize(referenceFrequencies.size(), nullptr);
		for (size_t k = 0; k < referenceFrequencies.size(); k++)
		{
			if ((m_frequencyCompoundingReferenceFrequencies[k] != referenceFrequencies[k] ||
				m_frequencyCompoundingBandwidths[k] != bandwidths[k]) &&
				referenceFrequencies[k] > 0 && bandwidths[k] > 0)
			{
				// the stored filter k does is not to given parameters -> create new
				m_frequencyCompoundingReferenceFrequencies[k] = referenceFrequencies[k];
				m_frequencyCompoundingBandwidths[k] = bandwidths[k];

				m_frequencyCompoundingBandpassFilters[k] =
					FirFilterFactory::createFilter<WorkType>(
						m_frequencyCompoundingBandpassFilterLength,
						FirFilterFactory::FilterTypeBandPass,
						FirFilterFactory::FilterWindowKaiser,
						m_samplingFrequency,
						m_frequencyCompoundingReferenceFrequencies[k],
						m_frequencyCompoundingBandwidths[k]);
				m_frequencyCompoundingBandpassFilters[k] = make_shared<Container<WorkType> >(LocationGpu, *m_frequencyCompoundingBandpassFilters[k]);
				bankNeedsUpdate = true;
			}

			if (m_frequencyCompoundingWeights[k] != weights[k])
			{
				m_frequencyCompoundingWeights[k] = weights[k];
				weightsNeedUpdate = true;
			}

			weightAcum += (WorkType)m_frequencyCompoundingWeights[k];
		}

		size_t numBandpassFilters = m_frequencyCompoundingBandpassFilters.size();

		//Copy the filters together for easy access
		if (!m_frequencyCompoundingBandpassFilterBank || m_frequencyCompoundingBandpassFilterBank->size() != numBandpassFilters * m_frequencyCompoundingBandpassFilterLength)
		{
			bankNeedsUpdate = true;
		}
		if (bankNeedsUpdate)
		{
			m_frequencyCompoundingReferenceFrequenciesOverSamplingFrequencies = make_shared<Container<WorkType> >(LocationHost, inImageData->getStream(), numBandpassFilters);
			m_frequencyCompoundingBandpassFilterBank = make_shared <Container<WorkType> >(LocationGpu, inImageData->getStream(), numBandpassFilters * m_frequencyCompoundingBandpassFilterLength);
			for (size_t k = 0; k < numBandpassFilters; k++)
			{
				cudaSafeCall(cudaMemcpy(
					m_frequencyCompoundingBandpassFilterBank->get() + k*m_frequencyCompoundingBandpassFilterLength,
					m_frequencyCompoundingBandpassFilters[k]->get(),
					sizeof(WorkType)* m_frequencyCompoundingBandpassFilterLength,
					cudaMemcpyDefault));

				m_frequencyCompoundingReferenceFrequenciesOverSamplingFrequencies->get()[k] = static_cast<WorkType>(
					m_frequencyCompoundingReferenceFrequencies[k] / m_samplingFrequency);
			}

			m_frequencyCompoundingReferenceFrequenciesOverSamplingFrequencies =
				make_shared<Container<WorkType> >(LocationGpu, *m_frequencyCompoundingReferenceFrequenciesOverSamplingFrequencies);
		}
		if (weightsNeedUpdate)
		{
			m_frequencyCompoundingWeightsGpu = make_shared<Container<WorkType> >(LocationHost, inImageData->getStream(), numBandpassFilters);
			for (size_t k = 0; k < numBandpassFilters; k++)
			{
				m_frequencyCompoundingWeightsGpu->get()[k] = static_cast<WorkType>(m_frequencyCompoundingWeights[k]);
			}
			m_frequencyCompoundingWeightsGpu = make_shared<Container<WorkType> >(LocationGpu, *m_frequencyCompoundingWeightsGpu);
		}

		// now that the filters are guaranteed to be current, perform frequency compounding envelope detection
		auto pBandpass = make_shared<Container<WorkType> >(LocationGpu, inImageData->getStream(), numScanlines*numSamples*numBandpassFilters);
		dim3 blockSizeFilter(16, 8);
		dim3 gridSizeFilter(
			static_cast<unsigned int>((numScanlines + blockSizeFilter.x - 1) / blockSizeFilter.x),
			static_cast<unsigned int>((numSamples + blockSizeFilter.y - 1) / blockSizeFilter.y));

		kernelFilterBankDemodulation<maxNumbandpassFilters, 0, 1, 3, 4> <<<gridSizeFilter, blockSizeFilter, 0, inImageData->getStream()>>> (
			inImageData->get(),
			m_frequencyCompoundingBandpassFilterBank->get(),
			pBandpass->get(),
			numSamples,
			numScanlines,
			(int)m_frequencyCompoundingBandpassFilterLength,
			(int)1,
			(int)numBandpassFilters,
			m_frequencyCompoundingWeightsGpu->get(),
			1/ weightAcum,
			m_frequencyCompoundingReferenceFrequenciesOverSamplingFrequencies->get());
		cudaSafeCall(cudaPeekAtLastError());

		//Apply the decimation lowpass filter on all bank outputs at the same time
		int numSamplesDecimated = decimatedSignalLength(numSamples, decimation);
		auto pEnv = make_shared<Container<OutputType> >(LocationGpu, inImageData->getStream(), numScanlines*numSamplesDecimated);

		uint32_t stride = std::max(decimation, (uint32_t)1);
		dim3 blockSizeDecimation(16, 8);
		dim3 gridSizeDecimation(
			static_cast<unsigned int>((numScanlines + blockSizeDecimation.x - 1) / blockSizeDecimation.x),
			static_cast<unsigned int>((numSamplesDecimated + blockSizeDecimation.y - 1) / blockSizeDecimation.y));

		kernelFilterStrided <<<gridSizeDecimation, blockSizeDecimation, 0, inImageData->getStream()>>> (
			pBandpass->get(),
			m_decimationLowpassFilter->get(),
			pEnv->get(),
			numSamples,
			numScanlines,
			(int)m_decimationLowpassFilterLength,
			stride);
		cudaSafeCall(cudaPeekAtLastError());

		return pEnv;
	}

	template 
	shared_ptr<Container<int16_t> > IQDemodulator::demodulateMagnitudeFrequencyCompounding<int16_t, int16_t>(
		const shared_ptr<const Container<int16_t> >& inImageData,
		int numScanlines, int numSamples, uint32_t decimation,
		const std::vector<double>& referenceFrequencies,
		const std::vector<double>& bandwidths,
		const std::vector<double>& weights);
	template
		shared_ptr<Container<int16_t> > IQDemodulator::demodulateMagnitudeFrequencyCompounding<float, int16_t>(
			const shared_ptr<const Container<float> >& inImageData,
			int numScanlines, int numSamples, uint32_t decimation,
			const std::vector<double>& referenceFrequencies,
			const std::vector<double>& bandwidths,
			const std::vector<double>& weights);
	template
		shared_ptr<Container<float> > IQDemodulator::demodulateMagnitudeFrequencyCompounding<int16_t, float>(
			const shared_ptr<const Container<int16_t> >& inImageData,
			int numScanlines, int numSamples, uint32_t decimation,
			const std::vector<double>& referenceFrequencies,
			const std::vector<double>& bandwidths,
			const std::vector<double>& weights);
	template
		shared_ptr<Container<float> > IQDemodulator::demodulateMagnitudeFrequencyCompounding<float, float>(
			const shared_ptr<const Container<float> >& inImageData,
			int numScanlines, int numSamples, uint32_t decimation,
			const std::vector<double>& referenceFrequencies,
			const std::vector<double>& bandwidths,
			const std::vector<double>& weights);
}