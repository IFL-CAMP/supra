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

#include "TimeGainCompensation.h"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

using namespace std;

namespace supra
{
	namespace TimeGainCompensationInternal
	{
		typedef TimeGainCompensation::WorkType WorkType;

		// here the actual processing happens!

		template <typename InputType, typename OutputType>
		__global__ void processKernel(const InputType* inputImage, vec3s size, size_t workDimension,
		        const WorkType* factors, OutputType* outputImage)
		{
			size_t x = blockDim.x*blockIdx.x + threadIdx.x;
			size_t y = blockDim.y*blockIdx.y + threadIdx.y;
			size_t z = blockDim.z*blockIdx.z + threadIdx.z;
			
			size_t width = size.x;
			size_t height = size.y;
			size_t depth = size.z;

			if (x < width && y < height && z < depth)
			{
				// Perform a pixel-wise operation on the image

				// Get the input pixel value and cast it to out working type.
				// As this should in general be a type with wider range / precision, this cast does not loose anything.
				WorkType inPixel = inputImage[x + y*width + z *width*height];

				// fetch factor according to workDimension
				WorkType factor = (WorkType)0;
				if (workDimension == 0)
                {
				    factor = factors[x];
                }
				else if (workDimension == 1)
                {
                    factor = factors[y];
                }
                else if (workDimension == 2)
                {
                    factor = factors[z];
                }

				// Perform operation, in this case multiplication
				WorkType value = inPixel * factor;

				// Store the output pixel value.
				// Because this is templated, we need to cast from "WorkType" to "OutputType".
				// This should happen in a sane way, that is with clamping. There is a helper for that!
				outputImage[x + y*width + z *width*height] = clampCast<OutputType>(value);
			}
		}
	}



	template <typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > TimeGainCompensation::process(
	        const shared_ptr<const Container<InputType>>& imageData, vec3s size, size_t workDimension)
	{
        assert(m_curvePoints.size() >= 2);
	    assert(workDimension <= 2);

		// here we prepare the buffers and call the cuda kernel
		size_t width = size.x;
		size_t height = size.y;
		size_t depth = size.z;

		size_t sizeWorkDimension = 0;
		switch(workDimension)
        {
            default:
		    case 0:
                sizeWorkDimension = size.x;
                break;
            case 1:
                sizeWorkDimension = size.y;
                break;
            case 2:
                sizeWorkDimension = size.z;
                break;
        }

        // Make sure the sampled curve is sampled to the right number of samples
        if (m_curveSampled == nullptr || m_curveSampled->size() != sizeWorkDimension)
        {
            sampleCurve(sizeWorkDimension);
        }

		// make sure the data is in cpu memory
		auto inImageData = imageData;
		if (!inImageData->isGPU() && !inImageData->isBoth())
		{
			inImageData = make_shared<Container<InputType> >(LocationGpu, *inImageData);
		}
		
		// prepare the output memory
		auto outImageData = make_shared<Container<OutputType> >(LocationGpu, inImageData->getStream(), width*height*depth);
		
		// call the kernel for the heavy-lifting
		dim3 blockSize(32, 4, 1);
		dim3 gridSize(
			static_cast<unsigned int>((size.x + blockSize.x - 1) / blockSize.x),
			static_cast<unsigned int>((size.y + blockSize.y - 1) / blockSize.y),
			static_cast<unsigned int>((size.z + blockSize.z - 1) / blockSize.z));
		TimeGainCompensationInternal::processKernel <<<gridSize, blockSize, 0, inImageData->getStream() >>> (
			inImageData->get(),
			size,
			workDimension,
			m_curveSampled->get(),
			outImageData->get());
		// check for cuda launch errors
		cudaSafeCall(cudaPeekAtLastError());
		// You should NOT synchronize the device or the stream we are working on!!

		// return the result!
		return outImageData;
	}

    void TimeGainCompensation::setCurve(const std::vector<std::pair<double, double> > &curvePoints)
    {
	    // We need at least two points for this to make sense
        assert(curvePoints.size() >= 2);

        m_curveSampled = nullptr;
        m_curvePoints = curvePoints;

        // Sort the curve points w.r.t. their depth
        std::sort(m_curvePoints.begin(), m_curvePoints.end(),
                [](const std::pair<double, double>& a, const std::pair<double, double>& b) { return a.first < b.first; });

        // If the first or last are not 0.0 or 1.0 respectively, add those points with the same weight as the adjacents
        assert(m_curvePoints[0].first >= 0.0);
        assert(m_curvePoints[m_curvePoints.size() - 1].first <= 1.0);
        if (m_curvePoints[0].first > 0.0)
        {
            m_curvePoints.insert(m_curvePoints.begin(), {0.0, m_curvePoints[0].second});
        }
        if (m_curvePoints[m_curvePoints.size() - 1].first < 1.0)
        {
            m_curvePoints.push_back({1.0, m_curvePoints[m_curvePoints.size() - 1].second});
        }
    }

    void TimeGainCompensation::sampleCurve(size_t numSamples)
    {
        std::unique_ptr<Container<WorkType> > curveSampledCpu = std::unique_ptr<Container<WorkType> >(
                new Container<WorkType>(LocationHost, cudaStreamDefault, numSamples));

        // Interpolate between values for each sample
        size_t currentCurvePoint = 0;
        for (size_t sample = 0; sample < numSamples; sample++)
        {
            double currentPosition = static_cast<double>(sample) / (numSamples - 1);
            if (currentPosition > m_curvePoints[currentCurvePoint + 1].first)
            {
                currentCurvePoint++;
            }
            double t = (currentPosition - m_curvePoints[currentCurvePoint].first) /
                    (m_curvePoints[currentCurvePoint + 1].first - m_curvePoints[currentCurvePoint].first);
            double levelInterpolated = (1.0 - t) * m_curvePoints[currentCurvePoint    ].second +
                                              t  * m_curvePoints[currentCurvePoint + 1].second;
            curveSampledCpu->get()[sample] = static_cast<WorkType>(pow(10.0, levelInterpolated / 20.0));
        }

        m_curveSampled = std::unique_ptr<Container<WorkType> >(new Container<WorkType>(LocationGpu, *curveSampledCpu));
    }

    // We don't wish to have the template implementation in the header, to make compilation easier.
	// Because of this, we need to explicity instantiate the methods we will need.
	template
	shared_ptr<Container<uint8_t> > TimeGainCompensation::process<int16_t, uint8_t>(const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, size_t workDimension);
	template
	shared_ptr<Container<uint8_t> > TimeGainCompensation::process<float, uint8_t>(const shared_ptr<const Container<float> >& inImageData, vec3s size, size_t workDimension);
	template
	shared_ptr<Container<uint8_t> > TimeGainCompensation::process<uint8_t, uint8_t>(const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size, size_t workDimension);
	template
	shared_ptr<Container<float> > TimeGainCompensation::process<int16_t, float>(const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, size_t workDimension);
	template
	shared_ptr<Container<float> > TimeGainCompensation::process<float, float>(const shared_ptr<const Container<float> >& inImageData, vec3s size, size_t workDimension);
	template
	shared_ptr<Container<float> > TimeGainCompensation::process<uint8_t, float>(const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size, size_t workDimension);
	template
	shared_ptr<Container<int16_t> > TimeGainCompensation::process<int16_t, int16_t>(const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, size_t workDimension);
	template
	shared_ptr<Container<int16_t> > TimeGainCompensation::process<float, int16_t>(const shared_ptr<const Container<float> >& inImageData, vec3s size, size_t workDimension);
	template
	shared_ptr<Container<int16_t> > TimeGainCompensation::process<uint8_t, int16_t>(const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size, size_t workDimension);
}