// ================================================================================================
// 
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License, version 2.1, as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this program.  If not, see
// <http://www.gnu.org/licenses/>.
//
// ================================================================================================

#include "NoiseCuda.h"

#include <curand.h>
#include <nppi_statistics_functions.h>
#include <nppi_filtering_functions.h>
#include <nppi_arithmetic_and_logical_operations.h>

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

namespace supra
{
	/// Verifies a curand call returned "CURAND_STATUS_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise.
	#define curandSafeCall(_err_) curandSafeCall2(_err_, __FILE__, __LINE__, FUNCNAME_PORTABLE)

	/// Verifies a curand call returned "CURAND_STATUS_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise. Calles by curandSafeCall
	inline bool curandSafeCall2(curandStatus err, const char* file, int line, const char* func) {

		//#ifdef CUDA_ERROR_CHECK
		if (CURAND_STATUS_SUCCESS != err) {
			char buf[1024];
			sprintf(buf, "cuRAND Error (in \"%s\", Line: %d, %s): %d\n", file, line, func, err);
			printf("%s", buf);
			logging::log_error(buf);
			return false;
		}

		//#endif
		return true;
	}

	/// Verifies a NPP call returned "NPP_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise.
	#define nppSafeCall(_err_) nppSafeCall2(_err_, __FILE__, __LINE__, FUNCNAME_PORTABLE)

	/// Verifies a NPP call returned "NPP_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise. Calles by nppSafeCall
	inline bool nppSafeCall2(NppStatus err, const char* file, int line, const char* func) {

		//#ifdef CUDA_ERROR_CHECK
		if (NPP_SUCCESS != err) {
			char buf[1024];
			sprintf(buf, "NPP Error (in \"%s\", Line: %d, %s): %d\n", file, line, func, err);
			printf("%s", buf);
			logging::log_error(buf);
			return false;
		}

		//#endif
		return true;
	}
	

	namespace NoiseCudaInternal
	{
		typedef NoiseCuda::WorkType WorkType;

		// here the actual processing happens!

		template <typename InputType, typename OutputType>
		__global__ void processKernel(
			const InputType* inputImage, 
			const WorkType* noiseAdditiveUniform,
			const WorkType* noiseAdditiveGauss,
			const WorkType* noiseMultiplicativeUniform,
			const WorkType* noiseMultiplicativeGauss,
			vec3s size,
			WorkType additiveUniformMin, WorkType additiveUniformRange,
			WorkType multiplicativeUniformMin, WorkType multiplicativeUniformRange, OutputType* outputImage)
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
				WorkType noiseAdditiveUniformVal = additiveUniformMin + additiveUniformRange * noiseAdditiveUniform[x + y * width + z * width*height];
				WorkType noiseAdditiveGaussVal = noiseAdditiveGauss[x + y * width + z * width*height];
				WorkType noiseMultiplicativeUniformVal = multiplicativeUniformMin + multiplicativeUniformRange * noiseMultiplicativeUniform[x + y * width + z * width*height];
				WorkType noiseMultiplicativeGaussVal = noiseMultiplicativeGauss[x + y * width + z * width*height];

				// Add the noise
				WorkType value = inPixel * noiseMultiplicativeUniformVal * noiseMultiplicativeGaussVal + noiseAdditiveUniformVal + noiseAdditiveGaussVal;

				// Store the output pixel value.
				// Because this is templated, we need to cast from "WorkType" to "OutputType".
				// This should happen in a sane way, that is with clamping. There is a helper for that!
				outputImage[x + y*width + z *width*height] = clampCast<OutputType>(value);
			}
		}
	}

	shared_ptr<Container<NoiseCuda::WorkType> > NoiseCuda::makeNoiseCorrelated(const shared_ptr<const Container<WorkType>>& in,
		size_t width, size_t height, size_t depth)
	{
		// Filter the noise to make it spatially correlated!
		int iWidth = static_cast<int>(width);
		int iHeight = static_cast<int>(height);
		int iDepth = static_cast<int>(depth);

		auto out = make_shared<Container<WorkType> >(LocationGpu, in->getStream(), width*height*depth);

		auto meanStdDevBefore = make_shared<Container<double> >(LocationGpu, in->getStream(), 2);
		auto meanStdDevAfter = make_shared<Container<double> >(LocationGpu, in->getStream(), 2);
		int scratchSize = 0;
		nppSafeCall(nppiMeanStdDevGetBufferHostSize_32f_C1R({ iHeight, iWidth*iDepth }, &scratchSize));
		auto meanStdDevScratch = make_shared<Container<uint8_t> >(LocationGpu, in->getStream(), scratchSize);

		nppSafeCall(nppiMean_StdDev_32f_C1R(
			in->get(),
			iHeight * sizeof(float),
			{ iHeight, iWidth*iDepth },
			meanStdDevScratch->get(),
			meanStdDevBefore->get(),
			meanStdDevBefore->get() + 1
		));
		cudaSafeCall(cudaDeviceSynchronize());
		for (size_t z = 0; z < depth; z++)
		{
			nppSafeCall(nppiFilterGaussBorder_32f_C1R(
				in->get() + z * width*height,
				iHeight * sizeof(float),
				{ iHeight, iWidth },
				{ 0, 0 },
				out->get() + z * width*height,
				iHeight * sizeof(float),
				{ iHeight, iWidth },
				//NPP_MASK_SIZE_5_X_5,
				NPP_MASK_SIZE_15_X_15,
				NPP_BORDER_REPLICATE
			));
			cudaSafeCall(cudaDeviceSynchronize());
		}
		nppSafeCall(nppiMean_StdDev_32f_C1R(
			out->get(),
			iHeight * sizeof(float),
			{ iHeight, iWidth*iDepth },
			meanStdDevScratch->get(),
			meanStdDevAfter->get(),
			meanStdDevAfter->get() + 1
		));
		cudaSafeCall(cudaDeviceSynchronize());

		auto meanStdDevBeforeHost = make_shared<Container<double> >(LocationHost, *meanStdDevBefore);
		auto meanStdDevAfterHost = make_shared<Container<double> >(LocationHost, *meanStdDevAfter);
		double meanBefore = meanStdDevBeforeHost->get()[0];
		double meanAfter = meanStdDevAfterHost->get()[0];
		double stdBefore = meanStdDevBeforeHost->get()[1];
		double stdAfter = meanStdDevAfterHost->get()[1];

		nppSafeCall(nppiSubC_32f_C1IR(
			static_cast<float>(meanAfter),
			out->get(),
			iHeight * sizeof(float),
			{ iHeight, iWidth*iDepth }));
		cudaSafeCall(cudaDeviceSynchronize());
		nppSafeCall(nppiMulC_32f_C1IR(
			static_cast<float>(stdBefore / stdAfter),
			out->get(),
			iHeight * sizeof(float),
			{ iHeight, iWidth*iDepth }));
		cudaSafeCall(cudaDeviceSynchronize());
		nppSafeCall(nppiAddC_32f_C1IR(
			static_cast<float>(meanBefore),
			out->get(),
			iHeight * sizeof(float),
			{ iHeight, iWidth*iDepth }));
		cudaSafeCall(cudaDeviceSynchronize());

		// (we need to sync before and after generation, as the npp kernels here are not launched on the same stream)
		//cudaSafeCall(cudaDeviceSynchronize());

		return out;
	}

	template <typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > NoiseCuda::process(const shared_ptr<const Container<InputType>>& imageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated)
	{
		// here we prepare the buffers and call the cuda kernel

		size_t width = size.x;
		size_t height = size.y;
		size_t depth = size.z;
		size_t numel = width * height * depth;

		// make sure the data is in cpu memory
		auto inImageData = imageData;
		if (!inImageData->isGPU() && !inImageData->isBoth())
		{
			inImageData = make_shared<Container<InputType> >(LocationGpu, *inImageData);
		}
		
		// prepare the output memory
		auto outImageData = make_shared<Container<OutputType> >(LocationGpu, inImageData->getStream(), numel);

		// prepare the noise 
		auto noiseAdditiveUniform = make_shared<Container<WorkType> >(LocationGpu, inImageData->getStream(), numel);
		auto noiseAdditiveGauss = make_shared<Container<WorkType> >(LocationGpu, inImageData->getStream(), numel);
		auto noiseMultiplicativeUniform = make_shared<Container<WorkType> >(LocationGpu, inImageData->getStream(), numel);
		auto noiseMultiplicativeGauss = make_shared<Container<WorkType> >(LocationGpu, inImageData->getStream(), numel);
		
		/* Create pseudo-random number generator */
		curandGenerator_t gen;
		//CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		curandSafeCall(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

		// Set seed
		//CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
		curandSafeCall(curandSetPseudoRandomGeneratorSeed(gen, rand()));

		// Make some noise!
		// (we need to sync before and after generation, as the cuRAND kernels here are not launched on the same stream)
		cudaSafeCall(cudaDeviceSynchronize());
		curandSafeCall(curandGenerateUniform(gen, noiseAdditiveUniform->get(), numel));
		curandSafeCall(curandGenerateNormal(gen, noiseAdditiveGauss->get(), numel, additiveGaussMean, additiveGaussStd));
		curandSafeCall(curandGenerateUniform(gen, noiseMultiplicativeUniform->get(), numel));
		curandSafeCall(curandGenerateNormal(gen, noiseMultiplicativeGauss->get(), numel, multiplicativeGaussMean, multiplicativeGaussStd));
		// (we need to sync before and after generation, as the cuRAND kernels here are not launched on the same stream)
		cudaSafeCall(cudaDeviceSynchronize());

		// Filter the noise to make it spatially correlated!
		if (additiveUniformCorrelated)
		{
			noiseAdditiveUniform = makeNoiseCorrelated(noiseAdditiveUniform, width, height, depth);
		}
		if (additiveGaussCorrelated)
		{
			noiseAdditiveGauss = makeNoiseCorrelated(noiseAdditiveGauss, width, height, depth);
		}
		if (multiplicativeUniformCorrelated)
		{
			noiseMultiplicativeUniform = makeNoiseCorrelated(noiseMultiplicativeUniform, width, height, depth);
		}
		if (multiplicativeGaussCorrelated)
		{
			noiseMultiplicativeGauss = makeNoiseCorrelated(noiseMultiplicativeGauss, width, height, depth);
		}
		// (we need to sync before and after generation, as the npp kernels here are not launched on the same stream)
		cudaSafeCall(cudaDeviceSynchronize());

		curandSafeCall(curandDestroyGenerator(gen));
		
		// call the kernel for the heavy-lifting
		dim3 blockSize(32, 4, 1);
		dim3 gridSize(
			static_cast<unsigned int>((size.x + blockSize.x - 1) / blockSize.x),
			static_cast<unsigned int>((size.y + blockSize.y - 1) / blockSize.y),
			static_cast<unsigned int>((size.z + blockSize.z - 1) / blockSize.z));
		NoiseCudaInternal::processKernel <<<gridSize, blockSize, 0, inImageData->getStream() >>> (
			inImageData->get(),
			noiseAdditiveUniform->get(), //noiseAdditiveUniform->get(),
			noiseAdditiveGauss->get(),
			noiseMultiplicativeUniform->get(),
			noiseMultiplicativeGauss->get(),
			size,
			additiveUniformMin, additiveUniformMax - additiveUniformMin,
			multiplicativeUniformMin, multiplicativeUniformMax - multiplicativeUniformMin,
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
		shared_ptr<Container<uint8_t> > NoiseCuda::process<int16_t, uint8_t>(const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, 
			WorkType additiveUniformMin, WorkType additiveUniformMax,
			WorkType additiveGaussMean, WorkType additiveGaussStd,
			WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
			WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
			bool additiveUniformCorrelated, bool additiveGaussCorrelated,
			bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	template
	shared_ptr<Container<uint8_t> > NoiseCuda::process<float, uint8_t>(const shared_ptr<const Container<float> >& inImageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	template
	shared_ptr<Container<uint8_t> > NoiseCuda::process<uint8_t, uint8_t>(const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	template
	shared_ptr<Container<float> > NoiseCuda::process<int16_t, float>(const shared_ptr<const Container<int16_t> >& inImageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	template
	shared_ptr<Container<float> > NoiseCuda::process<float, float>(const shared_ptr<const Container<float> >& inImageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	template
	shared_ptr<Container<float> > NoiseCuda::process<uint8_t, float>(const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	template
	shared_ptr<Container<int16_t> > NoiseCuda::process<int16_t, int16_t>(const shared_ptr<const Container<int16_t> >& inImageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	template
	shared_ptr<Container<int16_t> > NoiseCuda::process<float, int16_t>(const shared_ptr<const Container<float> >& inImageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	template
	shared_ptr<Container<int16_t> > NoiseCuda::process<uint8_t, int16_t>(const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size,
		WorkType additiveUniformMin, WorkType additiveUniformMax,
		WorkType additiveGaussMean, WorkType additiveGaussStd,
		WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
		WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
		bool additiveUniformCorrelated, bool additiveGaussCorrelated,
		bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
}