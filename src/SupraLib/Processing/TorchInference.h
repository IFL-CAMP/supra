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

#ifndef __TORCHINFERENCE_H__
#define __TORCHINFERENCE_H__

#ifdef HAVE_TORCH

#include <memory>

#include <torch/script.h>

#include <Container.h>
#include <vec.h>
#include <utilities/Logging.h>

namespace supra
{
	class TorchInference {
	public:
		TorchInference(
			const std::string& modelFilename, 
			const std::string& inputNormalization, 
			const std::string& outputDenormalization);

		template <typename InputType, typename OutputType>
		std::shared_ptr<ContainerBase> process(
			std::shared_ptr<const Container<InputType> >imageData,
			vec3s inputSize, vec3s outputSize,
			const std::string& currentLayout, const std::string& finalLayout,
			DataType modelInputDataType, DataType modelOutputDataType,
			const std::string& modelInputLayout, const std::string& modelOutputLayout,
			uint32_t inferencePatchSize, uint32_t inferencePatchOverlap)
		{
			assert(m_torchModule != nullptr);
			assert(m_inputNormalizationModule != nullptr);
			assert(m_outputDenormalizationModule != nullptr);

			std::shared_ptr <Container<OutputType> > pDataOut = nullptr;
			try {
				cudaSafeCall(cudaDeviceSynchronize());

				// Wrap our input data into a torch tensor (with first dimension 1, for batchsize 1)
				auto inputData = torch::from_blob((void*)(imageData->get()),
					{ (int64_t)1, (int64_t)inputSize.z, (int64_t)inputSize.y, (int64_t)inputSize.x },
					torch::TensorOptions().
					dtype(caffe2::TypeMeta::Make<InputType>()).
					device(imageData->isGPU() ? torch::kCUDA : torch::kCPU).
					requires_grad(false));

				if (inferencePatchSize == 0)
				{
					inferencePatchSize = inputSize.x;
				}
				assert(inferencePatchSize > inferencePatchOverlap * 2);

				size_t numPixels = inputSize.x;
				pDataOut = std::make_shared<Container<OutputType> >(LocationHost, imageData->getStream(), outputSize.x*outputSize.y*outputSize.z);

				size_t lastValidPixels = 0;
				for (size_t startPixelValid = 0; startPixelValid < numPixels; startPixelValid += lastValidPixels)
				{
					// Compute the size and position of the patch we want to run the model for
					size_t patchSizeValid = 0;
					size_t patchSize = 0;
					size_t startPixel = 0;
					if (startPixelValid == 0 && numPixels - startPixelValid <= inferencePatchSize)
					{
						//Special case: The requested patch size is large enough. No patching necessary!
						patchSize = numPixels - startPixelValid;
						patchSizeValid = patchSize;
						startPixel = 0;
					}
					else if (startPixelValid == 0)
					{
						// The first patch only needs to be padded on the bottom
						patchSize = inferencePatchSize;
						patchSizeValid = patchSize - inferencePatchOverlap;
						startPixel = 0;
					}
					else if (numPixels - (startPixelValid - inferencePatchOverlap) <= inferencePatchSize)
					{
						// The last patch only needs to be padded on the top
						startPixel = (startPixelValid - inferencePatchOverlap);
						patchSize = numPixels - startPixel;
						patchSizeValid = patchSize - inferencePatchOverlap;
					}
					else
					{
						// Every patch in the middle
						// padding on the top and bottom
						startPixel = (startPixelValid - inferencePatchOverlap);
						patchSize = inferencePatchSize;
						patchSizeValid = patchSize - 2 * inferencePatchOverlap;
					}
					lastValidPixels = patchSizeValid;

					// Slice the input data
                    cudaSafeCall(cudaDeviceSynchronize());
					auto inputDataPatch = inputData.slice(3, startPixel, startPixel + patchSize);
                    cudaSafeCall(cudaDeviceSynchronize());

					// Convert it to the desired input type
                    cudaSafeCall(cudaDeviceSynchronize());
					inputDataPatch = convertDataType(inputDataPatch, modelInputDataType);
                    cudaSafeCall(cudaDeviceSynchronize());

					// Adjust layout if necessary
					inputDataPatch = changeLayout(inputDataPatch, currentLayout, modelInputLayout);
					assert(!(inputDataPatch.requires_grad()));

					// Run model
					// Normalize the input
                    cudaSafeCall(cudaDeviceSynchronize());
					auto inputDataPatchIvalue = m_inputNormalizationModule->run_method("normalize", inputDataPatch);
                    cudaSafeCall(cudaDeviceSynchronize());
					
					// build module input data structure
					std::vector<torch::jit::IValue> inputs;
					inputs.push_back(inputDataPatchIvalue);

					// Execute the model and turn its output into a tensor.
                    cudaSafeCall(cudaDeviceSynchronize());
					auto result = m_torchModule->forward(inputs);
					cudaSafeCall(cudaDeviceSynchronize());
										
					// Denormalize the output
					result = m_outputDenormalizationModule->run_method("denormalize", result);
					at::Tensor output = result.toTensor();
                    output = output.to(torch::kFloat).to(torch::kCPU);
					// This should never happen right now.
					assert(!output.is_hip());

					// Adjust layout
					output = changeLayout(output, modelOutputLayout, finalLayout);

					// Copy to the result buffer (while converting to OutputType)
					auto outAccessor = output.accessor<float, 4>();

					// Determine which output dimension is affected by the patching
					auto permutation = layoutPermutation(modelOutputLayout, finalLayout);

					size_t outSliceStart = 0;
					size_t outSliceEnd = outputSize.z;
					size_t outSliceOffset = 0;
					size_t outLineStart = 0;
					size_t outLineEnd = outputSize.y;
					size_t outLineOffset = 0;
					size_t outPixelStart = 0;
					size_t outPixelEnd = outputSize.x;
					size_t outPixelOffset = 0;

					// Since the data is already permuted to the right layout for output, but we are working patchwise,
					// we need to restrict the affected dimension's indices
					if (permutation[1] == 3)
					{
						outSliceStart = startPixelValid;
						outSliceEnd = startPixelValid + patchSizeValid;
						outSliceOffset = startPixel;
					}
					else if (permutation[2] == 3)
					{
						outLineStart = startPixelValid;
						outLineEnd = startPixelValid + patchSizeValid;
						outLineOffset = startPixel;
					}
					else if (permutation[3] == 3)
					{
						outPixelStart = startPixelValid;
						outPixelEnd = startPixelValid + patchSizeValid;
						outPixelOffset = startPixel;
					}
					
					for (size_t outSliceIdx = outSliceStart; outSliceIdx < outSliceEnd; outSliceIdx++)
					{
						for (size_t outLineIdx = outLineStart; outLineIdx < outLineEnd; outLineIdx++)
						{
							for (size_t outPixelIdx = outPixelStart; outPixelIdx < outPixelEnd; outPixelIdx++)
							{
								pDataOut->get()[
										outSliceIdx * outputSize.y * outputSize.x +
										outLineIdx * outputSize.x + 
										outPixelIdx] =
									clampCast<OutputType>(outAccessor[0]
										[outSliceIdx - outSliceOffset]
										[outLineIdx - outLineOffset]
										[outPixelIdx - outPixelOffset]);
							}
						}
					}
				}
				cudaSafeCall(cudaDeviceSynchronize());
			}
			catch (c10::Error e)
			{
				logging::log_error("TorchInference: Error (c10::Error) while running model '", m_modelFilename, "'");
				logging::log_error("TorchInference: ", e.what());
				logging::log_error("TorchInference: ", e.msg_stack());
			}
			catch (std::runtime_error e)
			{
				logging::log_error("TorchInference: Error (std::runtime_error) while running model '", m_modelFilename, "'");
				logging::log_error("TorchInference: ", e.what());
			}

			return pDataOut;
		}

	private:
		void loadModule();
		at::Tensor convertDataType(at::Tensor tensor, DataType datatype);
		at::Tensor changeLayout(at::Tensor tensor, const std::string& currentLayout, const std::string& outLayout);
		std::vector<int64_t> layoutPermutation(const std::string& currentLayout, const std::string& outLayout);

		std::shared_ptr<torch::jit::script::Module> m_torchModule;
		std::shared_ptr<torch::jit::script::Module> m_inputNormalizationModule;
		std::shared_ptr<torch::jit::script::Module> m_outputDenormalizationModule;

		std::string m_modelFilename;
		std::string m_inputNormalization;
		std::string m_outputDenormalization;
	};
}

#endif //HAVE_TORCH

#endif //!__TORCHINFERENCE_H__
