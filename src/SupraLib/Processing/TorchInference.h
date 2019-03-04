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

// forward declaration
//namespace torch 
//{
//	namespace jit 
//	{
//		namespace script 
//		{
//			struct Module;
//		}
//	}
//}
//namespace at
//{
//	class Tensor;
//}

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
			shared_ptr <Container<OutputType> > pDataOut = nullptr;
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

				// TODO rename
				size_t numSamples = inputSize.x;
				size_t numScanlines = inputSize.z;
				pDataOut = make_shared<Container<OutputType> >(LocationHost, imageData->getStream(), outputSize.x*outputSize.y*outputSize.z);

				size_t lastValidSamples = 0;
				for (size_t startSampleValid = 0; startSampleValid < numSamples; startSampleValid += lastValidSamples)
				{
					// Compute the size and position of the patch we want to run the model for
					size_t sliceSizeValid = 0;
					size_t sliceSize = 0;
					size_t startSample = 0;
					if (startSampleValid == 0 && numSamples - startSampleValid <= inferencePatchSize)
					{
						//Special case: The requested slice size is large enough. No patching necessary!
						sliceSize = numSamples - startSampleValid;
						sliceSizeValid = sliceSize;
						startSample = 0;
					}
					else if (startSampleValid == 0)
					{
						// The first patch only needs to be padded on the bottom
						sliceSize = inferencePatchSize;
						sliceSizeValid = sliceSize - inferencePatchOverlap;
						startSample = 0;
					}
					else if (numSamples - (startSampleValid - inferencePatchOverlap) <= inferencePatchSize)
					{
						// The last patch only needs to be padded on the top
						startSample = (startSampleValid - inferencePatchOverlap);
						sliceSize = numSamples - startSample;
						sliceSizeValid = sliceSize - inferencePatchOverlap;
					}
					else
					{
						// Every patch in the middle
						// padding on the top and bottom
						startSample = (startSampleValid - inferencePatchOverlap);
						sliceSize = inferencePatchSize;
						sliceSizeValid = sliceSize - 2 * inferencePatchOverlap;
					}
					lastValidSamples = sliceSizeValid;
					logging::log_always("sliceSizeValid: ", sliceSizeValid, " startSampleValid: ", startSampleValid);

					// Slice the input data
					auto inputDataSlice = inputData.slice(3, startSample, startSample + sliceSize);

					// Convert it to the desired input type
					inputDataSlice = convertDataType(inputDataSlice, modelInputDataType);

					// Adjust layout if necessary
					inputDataSlice = changeLayout(inputDataSlice, currentLayout, modelInputLayout);
					assert(!(inputDataSlice.requires_grad()));

					// Run model
					// Normalize the input
					auto inputDataSliceIvalue = m_inputNormalizationModule->run_method("normalize", inputDataSlice);
					
					// build module input data structure
					std::vector<torch::jit::IValue> inputs;
					inputs.push_back(inputDataSliceIvalue);

					// Execute the model and turn its output into a tensor.
					auto result = m_torchModule->forward(inputs);
					cudaSafeCall(cudaDeviceSynchronize());
										
					// Denormalize the output
					result = m_outputDenormalizationModule->run_method("denormalize", result);
					at::Tensor output = result.toTensor();
					// This should never happen right now.
					assert(!output.is_hip());

					// Adjust layout
					output = changeLayout(output, modelOutputLayout, finalLayout);

					// Copy to the result buffer (while converting to OutputType)
					output = output.to(torch::kCPU);
					logging::log_always("TORCH: out ndim: ", output.ndimension(), ", ", output.sizes());
					auto outAccessor = output.accessor<float, 4>();
					size_t sampleOffset = startSampleValid - startSample;

					//TODO
					//for z in outSize.z
					//	for y in outSize.y
					//		for x in patchX...
					for (size_t sampleIdxLocal = 0; sampleIdxLocal < sliceSizeValid; sampleIdxLocal++)
					{
						for (size_t scanlineIdx = 0; scanlineIdx < numScanlines; scanlineIdx++)
						{
							pDataOut->get()[(startSampleValid + sampleIdxLocal) * numScanlines + scanlineIdx] = 
								clampCast<OutputType>(outAccessor[0][0][sampleIdxLocal + sampleOffset][scanlineIdx]);
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
