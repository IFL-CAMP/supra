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

#include "TemporalFilter.h"

#include <assert.h>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

using namespace std;

namespace supra
{
	template <typename InputType, typename OutputType, typename WorkType>
	__global__ void temporalFilterKernel(
		size_t numel,
		uint32_t numImages,
		const double* __restrict__ weights,
		const InputType* __restrict__ * __restrict__ inputImages,
		OutputType* __restrict__ out)
	{
		size_t linearIdx = blockDim.x * blockIdx.x + threadIdx.x;
		if (linearIdx < numel)
		{
			WorkType sum = 0;
			WorkType sumWeights = 0;
			for (uint32_t imageIdx = 0; imageIdx < numImages; imageIdx++)
			{
				WorkType weight = static_cast<WorkType>(weights[imageIdx]);
				sum += weight * static_cast<WorkType>(inputImages[imageIdx][linearIdx]);
				sumWeights += weight;
			}
			sum /= sumWeights;
			out[linearIdx] = static_cast<OutputType>(sum);
		}
	}

	template<>
	shared_ptr<Container<int16_t> > TemporalFilter::filter(
		const std::queue<std::shared_ptr<const Container<int16_t> > > & inImageData,
		vec3s size,
		const std::vector<double> weights)
	{
		assert(inImageData.size() == weights.size());

		if (inImageData.size() == 1)
		{
			return make_shared<Container<int16_t> >(LocationGpu, *(inImageData.front()));
		}

		size_t width = size.x;
		size_t height = size.y;
		size_t depth = size.z;
		size_t numel = width*height*depth;

		auto stream = inImageData.back()->getStream();
		auto pFiltGpu = make_shared<Container<int16_t> >(LocationGpu, stream, numel);

		Container<double> weightsContainer(LocationGpu, stream, weights);
		Container<const int16_t*> imagePointersContainer(LocationHost, stream, inImageData.size());
		vector<shared_ptr<const Container<int16_t> > > copiedImages;
		queue<shared_ptr<const Container<int16_t> > > imageData = inImageData;
		for (size_t imageIndex = 0; imageIndex < inImageData.size(); imageIndex++)
		{
			auto thisImageData = imageData.front();
			imageData.pop();
			if (thisImageData->isGPU())
			{
				imagePointersContainer.get()[imageIndex] = thisImageData->get();
			}
			else
			{
				auto copy = make_shared<Container<int16_t> >(LocationHost, *thisImageData);
				copiedImages.push_back(copy);
				imagePointersContainer.get()[imageIndex] = copy->get();
			}
		}
		auto imagePointersContainerGpu = make_shared<Container<const int16_t*> >(LocationGpu, imagePointersContainer);

		dim3 blockSize(256, 1);
		dim3 gridSize(static_cast<unsigned int>((numel + blockSize.x - 1) / blockSize.x), 1);

		temporalFilterKernel<int16_t, int16_t, WorkType> <<<gridSize, blockSize, 0, stream>>> (
			numel,
			static_cast<uint32_t>(inImageData.size()),
			weightsContainer.get(),
			imagePointersContainerGpu->get(),
			pFiltGpu->get());
		cudaSafeCall(cudaPeekAtLastError());

		return pFiltGpu;
	}
}