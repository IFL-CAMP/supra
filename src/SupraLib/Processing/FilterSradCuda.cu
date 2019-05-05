// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Walter Simson 
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "FilterSradCuda.h"

#include "utilities/Buffer.h"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

using namespace std;

namespace supra
{
	namespace FilterSradCudaInternal
	{
		typedef FilterSradCuda::WorkType WorkType;

		struct thrustAdd {
			thrustAdd(WorkType _summand) : summand(_summand) {};
			WorkType summand;

			__host__ __device__ WorkType operator()(const WorkType& a) const
			{
				return a + summand;
			}
		};
	
		__device__ WorkType computeC(const Buffer3<const WorkType*, size_t>& imageBuffer, const vec3s& position, const vec3s& size, const WorkType& speckleScaleCurrentSq, const WorkType& eps)
		{
			size_t width = size.x;
			size_t height = size.y;
			//size_t depth = size.z;

			// Get the input pixel value and cast it to out working type.
			WorkType inPixel_c = imageBuffer[position];
			WorkType inPixel_t = ((position.y + 1) == height) ? inPixel_c : imageBuffer[position + vec3s{ 0, 1, 0 }];
			WorkType inPixel_b = (position.y == 0) ? inPixel_c : imageBuffer[position - vec3s{ 0, 1, 0 }];
			WorkType inPixel_l = (position.x == 0) ? inPixel_c : imageBuffer[position - vec3s{ 1, 0, 0 }];
			WorkType inPixel_r = ((position.x + 1) == width) ? inPixel_c : imageBuffer[position + vec3s{ 1, 0, 0 }];
			// TODO: 3D-ness

			// Calculate derivatives and Laplacian
			// Equations 52, 53, 54 respectively; h=1
			vec2T<WorkType> gradient_f{ inPixel_r - inPixel_c, inPixel_b - inPixel_c };
			vec2T<WorkType> gradient_b{ inPixel_c - inPixel_l, inPixel_c - inPixel_t };

			WorkType gradNormSq = (squ(gradient_f.x) + squ(gradient_f.y) + squ(gradient_b.x) + squ(gradient_b.y)) / squ(inPixel_c);
			WorkType laplacian = inPixel_r + inPixel_l + inPixel_b + inPixel_t - 4 * inPixel_c;

			// calculate diffusion coefficient
			// Eq. 57 (see Eq. 35)
			WorkType numerator = 0.5f * gradNormSq - (1 / 16)*squ(laplacian / inPixel_c);
			WorkType denominator = squ(1 + 0.25f * (laplacian / inPixel_c)) + eps;

			WorkType instCoefVarSq = numerator / denominator; // we only use the squared term i.e. no root

			// Calculate c (Eq. 33)
			WorkType diff_coef = 1.0f / (1.0f + (instCoefVarSq - speckleScaleCurrentSq) / (speckleScaleCurrentSq * (1.0f + speckleScaleCurrentSq)));

			// clamp c
			diff_coef = min(max(diff_coef, (WorkType)0), (WorkType)1);

			return diff_coef;
		}

		// here the actual processing happens!
		__global__ void processKernel(const WorkType* image, vec3s size, WorkType eps,
			WorkType lambda,
			WorkType speckleScale,
			WorkType* scratch)
		{
			size_t x = blockDim.x*blockIdx.x + threadIdx.x;
			size_t y = blockDim.y*blockIdx.y + threadIdx.y;
			size_t z = blockDim.z*blockIdx.z + threadIdx.z;

			Buffer3<const WorkType*, size_t> imageBuffer{image, size};
			Buffer3<WorkType*, size_t> scratchBuffer{ scratch, size };
			
			size_t width = size.x;
			size_t height = size.y;
			size_t depth = size.z;

			if (x < width && y < height && z < depth)
			{
				// Calculate Speckle Scale of Current Step Squared. Eq. (37).
				WorkType speckleScaleCurrentSq = squ(speckleScale);

				vec3s position{ x, y, z };

				auto c_c = computeC(imageBuffer, position, size, speckleScaleCurrentSq, eps);
				auto c_b = ((y + 1) == height) ? c_c : computeC(imageBuffer, position + vec3s{ 0, 1, 0 }, size, speckleScaleCurrentSq, eps);
				auto c_r = ((x + 1) == width) ? c_c : computeC(imageBuffer, position + vec3s{ 1, 0, 0 }, size, speckleScaleCurrentSq, eps);

				// Get the input pixel value and cast it to out working type.
				WorkType inPixel_c = imageBuffer[position];
				WorkType inPixel_t = ((position.y + 1) == height) ? inPixel_c : imageBuffer[position + vec3s{ 0, 1, 0 }];
				WorkType inPixel_b = (position.y == 0) ? inPixel_c : imageBuffer[position - vec3s{ 0, 1, 0 }];
				WorkType inPixel_l = (position.x == 0) ? inPixel_c : imageBuffer[position - vec3s{ 1, 0, 0 }];
				WorkType inPixel_r = ((position.x + 1) == width) ? inPixel_c : imageBuffer[position + vec3s{ 1, 0, 0 }];
				// TODO: 3D-ness

				// Calculate derivatives and Laplacian
				// Equations 52, 53, 54 respectively; h=1
				vec2T<WorkType> gradient_f{ inPixel_r - inPixel_c, inPixel_b - inPixel_c };
				vec2T<WorkType> gradient_b{ inPixel_c - inPixel_l, inPixel_c - inPixel_t };

				// compute the divergence of the c*grad(I) (58)
				WorkType d = c_r * gradient_f.x - c_c * gradient_b.x + c_b * gradient_f.y - c_c * gradient_b.y;

				scratchBuffer[position] = inPixel_c + lambda * 0.25f * d;
			}
		}
	}

	template <typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > FilterSradCuda::process(
		const shared_ptr<const Container<InputType>>& imageData, vec3s size, 
		double eps,  uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay)
	{
		// here we prepare the buffers and call the cuda kernel
		size_t width = size.x;
		size_t height = size.y;
		size_t depth = size.z;

		// make sure the data is in gpu memory
		auto inImageData = imageData;
		if (!inImageData->isGPU() && !inImageData->isBoth())
		{
			inImageData = make_shared<Container<InputType> >(LocationGpu, *inImageData);
		}
		
		// prepare the output memory
		auto outImageData = make_shared<Container<OutputType> >(LocationGpu, inImageData->getStream(), width*height*depth);
		// prepate the scratch memory
		auto currentBuffer = make_shared<Container<WorkType> >(LocationGpu, inImageData->getStream(), width*height*depth);
		auto nextBuffer = make_shared<Container<WorkType> >(LocationGpu, inImageData->getStream(), width*height*depth);

		// Copy the image to the first scratch buffer and cast it to WorkType
		thrust::transform(thrust::cuda::par.on(inImageData->getStream()),
			inImageData->get(),
			inImageData->get() + width * height*depth,
			currentBuffer->get(),
			clampCaster<WorkType, InputType>());
		thrust::transform(thrust::cuda::par.on(inImageData->getStream()),
			currentBuffer->get(),
			currentBuffer->get() + width * height*depth,
			currentBuffer->get(),
			FilterSradCudaInternal::thrustAdd((WorkType)1));
		cudaSafeCall(cudaPeekAtLastError());
		//cudaSafeCall(cudaDeviceSynchronize());
		
		// call the kernel for the heavy-lifting
		dim3 blockSize(32, 4, 1);
		dim3 gridSize(
			static_cast<unsigned int>((size.x + blockSize.x - 1) / blockSize.x),
			static_cast<unsigned int>((size.y + blockSize.y - 1) / blockSize.y),
			static_cast<unsigned int>((size.z + blockSize.z - 1) / blockSize.z));

		// Loop over the time steps of the solver
		for (uint32_t i = 0; i < numberIterations; i++)
		{
			WorkType speckleScaleCurrent = static_cast<WorkType>(speckleScale * std::exp(-speckleScaleDecay * lambda*i));

			FilterSradCudaInternal::processKernel <<<gridSize, blockSize, 0, inImageData->getStream() >>> (
				currentBuffer->get(),
				size,
				static_cast<WorkType>(eps),
				static_cast<WorkType>(lambda),
				speckleScaleCurrent,
				nextBuffer->get());
			// check for cuda launch errors
			cudaSafeCall(cudaPeekAtLastError());

			std::swap(nextBuffer, currentBuffer);
		}

		// Copy and cast the image from scratch (in WorkType) to the output buffer
		thrust::transform(thrust::cuda::par.on(inImageData->getStream()),
			currentBuffer->get(),
			currentBuffer->get() + width * height*depth,
			currentBuffer->get(),
			FilterSradCudaInternal::thrustAdd((WorkType)-1));
		thrust::transform(thrust::cuda::par.on(inImageData->getStream()), 
			currentBuffer->get(), 
			currentBuffer->get() + width * height*depth, 
			outImageData->get(), 
			clampCaster<OutputType, WorkType>());
		cudaSafeCall(cudaPeekAtLastError());

		// return the result!
		return outImageData;
	}

	// We don't wish to have the template implementation in the header, to make compilation easier.
	// Because of this, we need to explicity instantiate the methods we will need.
	template
	shared_ptr<Container<uint8_t> > FilterSradCuda::process<int16_t, uint8_t>(
		const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
	template
	shared_ptr<Container<uint8_t> > FilterSradCuda::process<float, uint8_t>(
		const shared_ptr<const Container<float> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
	template
	shared_ptr<Container<uint8_t> > FilterSradCuda::process<uint8_t, uint8_t>(
		const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
	template
	shared_ptr<Container<float> > FilterSradCuda::process<int16_t, float>(
		const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
	template
	shared_ptr<Container<float> > FilterSradCuda::process<float, float>(
		const shared_ptr<const Container<float> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
	template
	shared_ptr<Container<float> > FilterSradCuda::process<uint8_t, float>(
		const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
	template
	shared_ptr<Container<int16_t> > FilterSradCuda::process<int16_t, int16_t>(
		const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
	template
	shared_ptr<Container<int16_t> > FilterSradCuda::process<float, int16_t>(
		const shared_ptr<const Container<float> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
	template
	shared_ptr<Container<int16_t> > FilterSradCuda::process<uint8_t, int16_t>(
		const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size, 
		double eps, uint32_t numberIterations, double lambda, double speckleScale, double speckleScaleDecay);
}