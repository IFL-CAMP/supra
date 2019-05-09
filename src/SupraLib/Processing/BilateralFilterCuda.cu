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

#include "BilateralFilterCuda.h"
#include "utilities/Buffer.h"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

using namespace std;

namespace supra
{
	namespace BilateralFilterCudaInternal
	{
		typedef BilateralFilterCuda::WorkType WorkType;

		// here the actual processing happens!

		template <typename InputType, typename OutputType>
		__global__ void processKernel(const InputType* inputImage, const vec3s size, 
									const vec3T<WorkType> sigmaSpatial, const vec3s filterSize,
									WorkType sigmaItensity, OutputType* outputImage)
		{
			size_t x = blockDim.x*blockIdx.x + threadIdx.x;
			size_t y = blockDim.y*blockIdx.y + threadIdx.y;
			size_t z = blockDim.z*blockIdx.z + threadIdx.z;
			
			size_t width = size.x;
			size_t height = size.y;
			size_t depth = size.z;

			// declare the shared memory with dynamic size
			extern __shared__ uint8_t smem[];

			// create the buffer objects
			// These wrap the image pointers and allow for easy indexing and caching (here for inputBuffer)
			CachedBuffer3<const InputType*, size_t> inputBuffer{
				inputImage, size, reinterpret_cast<InputType*>(smem), vec3s{blockDim.x, blockDim.y, blockDim.z},
				vec3s{blockDim.x*blockIdx.x, blockDim.y*blockIdx.y, blockDim.z*blockIdx.z } };
			Buffer3<OutputType*, size_t> outputBuffer{ outputImage, size };

			if (x < width && y < height && z < depth)
			{
				// Perform the computations for one output-pixel
				// -> loop over the spatial window
				vec3s indexStart{
					max(x, filterSize.x / 2) - filterSize.x / 2,
					max(y, filterSize.y / 2) - filterSize.y / 2,
					max(z, filterSize.z / 2) - filterSize.z / 2 };
				vec3s indexEnd{
					min(x + filterSize.x / 2, width - 1),
					min(y + filterSize.y / 2, height - 1),
					min(z + filterSize.z / 2, depth - 1) };
				WorkType accumulatedWeight = 0;
				WorkType valueFiltered = 0;
				WorkType valueCenter = inputBuffer[{x, y, z}];
				vec3s index{ indexStart };
				for (; index.z <= indexEnd.z; index.z++)
				{
					index.y = indexStart.y;
					for (; index.y <= indexEnd.y; index.y++)
					{
						index.x = indexStart.x;
						for (; index.x <= indexEnd.x; index.x++)
						{
							WorkType valueOffset = inputBuffer[index];
							// Compute the weight for this pixel in the window
							// in this case for the bilateral filter
							WorkType weight = 
								exp(
									// spatial term
									- (squ(index.x - x) / (2*squ(sigmaSpatial.x))
										+ squ(index.y - y) / (2*squ(sigmaSpatial.y))
										+ squ(index.z - z) / (2*squ(sigmaSpatial.z)))
									// intensity term
									- squ(valueOffset - valueCenter) / (2*squ(sigmaItensity))
								);
							// Update the result
							valueFiltered += weight * valueOffset;
							// remember used weights for later normalization
							accumulatedWeight += weight;
						}
					}
				}

				// normalize result w.r.t. employed weights
				if (accumulatedWeight != 0)
				{
					valueFiltered /= accumulatedWeight;
				}

				// Store the output pixel value.
				// Because this is templated, we need to cast from "WorkType" to "OutputType".
				// This should happen in a sane way, that is with clamping. There is a helper for that!
				outputBuffer[{x, y, z}] = clampCast<OutputType>(valueFiltered);
			}
		}
	}

	template <typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > BilateralFilterCuda::process(
		const shared_ptr<const Container<InputType>>& imageData, vec3s size, 
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity)
	{
		// here we prepare the buffers and call the cuda kernel

		size_t width = size.x;
		size_t height = size.y;
		size_t depth = size.z;

		// precompute parameters
		vec3s filterSize = static_cast<vec3s>(ceil(sigmaSpatialPixels * 2) * 2 + 1);

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
		// since we will use a caching buffer object inside the kernel, that places parts of the input 
		// in shared memory, we have to specify the size of that
		size_t sharedMemorySize = blockSize.x * blockSize.y * blockSize.z * sizeof(InputType);
		BilateralFilterCudaInternal::processKernel <<<gridSize, blockSize, sharedMemorySize, inImageData->getStream() >>> (
			inImageData->get(),
			size,
			sigmaSpatialPixels, 
			filterSize,
			sigmaItensity,
			outImageData->get());
		// check for cuda launch errors
		cudaSafeCall(cudaPeekAtLastError());
		// You should NOT synchronize the device or the stream we are working on!!

		// return the result!
		return outImageData;
	}

	// We don't wish to have the template implementation in the header, to make compilation easier.
	// Because of this, we need to explicity instantiate the methods we will need.
	template
	shared_ptr<Container<uint8_t> > BilateralFilterCuda::process<int16_t, uint8_t>(
		const shared_ptr<const Container<int16_t> >& imageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
	template
	shared_ptr<Container<uint8_t> > BilateralFilterCuda::process<float, uint8_t>(
		const shared_ptr<const Container<float> >& imageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
	template
	shared_ptr<Container<uint8_t> > BilateralFilterCuda::process<uint8_t, uint8_t>(
		const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
	template
	shared_ptr<Container<float> > BilateralFilterCuda::process<int16_t, float>(
		const shared_ptr<const Container<int16_t> >& inImageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
	template
	shared_ptr<Container<float> > BilateralFilterCuda::process<float, float>(
		const shared_ptr<const Container<float> >& inImageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
	template
	shared_ptr<Container<float> > BilateralFilterCuda::process<uint8_t, float>(
		const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
	template
	shared_ptr<Container<int16_t> > BilateralFilterCuda::process<int16_t, int16_t>(
		const shared_ptr<const Container<int16_t> >& inImageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
	template
	shared_ptr<Container<int16_t> > BilateralFilterCuda::process<float, int16_t>(
		const shared_ptr<const Container<float> >& inImageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
	template
	shared_ptr<Container<int16_t> > BilateralFilterCuda::process<uint8_t, int16_t>(
		const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size,
		const vec3T<WorkType>& sigmaSpatialPixels, WorkType sigmaItensity);
}