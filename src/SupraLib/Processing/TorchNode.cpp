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

#include "TorchNode.h"

#include <torch/script.h>

#include "USImage.h"
#include "Beamformer/USRawData.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	TorchNode::TorchNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_torchModule{ nullptr }
	{
		// Create the underlying tbb node for handling the message passing. This usually does not need to be modified.
		if (queueing)
		{
			m_node = unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndProcess(inObj); }));
		}
		else
		{
			m_node = unique_ptr<NodeTypeDiscarding>(
				new NodeTypeDiscarding(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndProcess(inObj); }));
		}

		m_callFrequency.setName("TorchNode");

		// Define the parameters that this node reveals to the user
		m_valueRangeDictionary.set<string>("modelFilename", "model.pt", "Model filename");
		
		// read the configuration to apply the default values
		configurationChanged();
	}

	void TorchNode::configurationChanged()
	{
		m_modelFilename = m_configurationDictionary.get<string>("modelFilename");
		loadModule();
	}

	void TorchNode::configurationEntryChanged(const std::string& configKey)
	{
		// lock the object mutex to make sure no processing happens during parameter changes
		unique_lock<mutex> l(m_mutex);
		if (configKey == "modelFilename")
		{
			m_modelFilename = m_configurationDictionary.get<string>("modelFilename");
			loadModule();
		}
	}

	template <typename InputType>
	std::shared_ptr<RecordObject> TorchNode::processTemplateSelection(
	        std::shared_ptr<const Container<InputType> > imageData, 
			vec3s size, 
			size_t workDimension, 
			std::shared_ptr<const USImageProperties> pImageProperties)
	{
		// With the function already templated on the input type, handle the desired output type.
		shared_ptr<RecordObject> pOut = nullptr;

		//switch (m_outputType)
		{
			try {
				cudaSafeCall(cudaDeviceSynchronize());

				// Wrap our input data into a tensor (with first dimension 1, for batchsize 1)
				auto inputData = torch::from_blob((void*)(imageData->get()), 
					{(int64_t)1, (int64_t)size.z, (int64_t)size.y, (int64_t)size.x }, 
					torch::TensorOptions().
						dtype(caffe2::TypeMeta::Make<InputType>()).
						device(imageData->isGPU() ? torch::kCUDA : torch::kCPU).
						requires_grad(false));

				size_t sliceSizeMax = 100;
				size_t sliceOverlap = 15; // On each side
				assert(sliceSizeMax > sliceOverlap * 2);

				vec2s outSize{ size.x, size.z };
				size_t numSamples = size.x;
				size_t numScanlines = size.z;
				shared_ptr<Container<float> > pDataOut = make_shared<Container<float> >(LocationHost, imageData->getStream(), outSize.x*outSize.y);

				size_t lastValidSamples = 0;
				for (size_t startSampleValid = 0; startSampleValid < numSamples; startSampleValid += lastValidSamples)
				{
					size_t sliceSizeValid = 0;
					size_t sliceSize = 0;
					size_t startSample = 0;
					if (startSampleValid == 0 && numSamples - startSampleValid <= sliceSizeMax)
					{
						//Special case: The requested slice size is large enough. No patching necessary!
						sliceSize = numSamples - startSampleValid;
						sliceSizeValid = sliceSize;
						startSample = 0;
					}
					else if (startSampleValid == 0)
					{
						// The first patch only needs to be padded on the bottom
						sliceSize = sliceSizeMax;
						sliceSizeValid = sliceSize - sliceOverlap;
						startSample = 0;
					}
					else if (numSamples - (startSampleValid - sliceOverlap) <= sliceSizeMax)
					{
						// The last patch only needs to be padded on the top
						startSample = (startSampleValid - sliceOverlap);
						sliceSize = numSamples - startSample;
						sliceSizeValid = sliceSize - sliceOverlap;
					}
					else
					{
						// Every patch in the middle
						// padding on the top and bottom
						startSample = (startSampleValid - sliceOverlap);
						sliceSize = sliceSizeMax;
						sliceSizeValid = sliceSize - 2*sliceOverlap;
					} 
					lastValidSamples = sliceSizeValid;
					logging::log_always("sliceSizeValid: ", sliceSizeValid, " startSampleValid: ", startSampleValid);
					
					auto inputDataSlice = inputData.slice(3, startSample, startSample + sliceSize);
					inputDataSlice = inputDataSlice.to(torch::kFloat);

					// normalize
					inputDataSlice = inputDataSlice.add(2047.0f).mul(1.0f / (2 * 2047));

					// BATCH X C X W X H
					//auto inputDataOnes = torch::ones({ 1, (int64_t)size.z, 64, 100 });
					//inputDataOnes = inputDataOnes.to(torch::kCUDA);
					//inputDataOnes.set_requires_grad(false);
					inputDataSlice = inputDataSlice.permute({ 0, 2, 1, 3 });
					assert(!(inputDataSlice.requires_grad()));

					std::vector<torch::jit::IValue> inputs;
					inputs.push_back(inputDataSlice);

					// Execute the model and turn its output into a tensor.
					auto result = m_torchModule->forward(inputs);
					at::Tensor output = result.toTensor().to(torch::kCPU);
					cudaSafeCall(cudaDeviceSynchronize());

					// denormalize
					output = output.mul(255.0f);

					// This should never happen right now.
					assert(!output.is_hip());

					//output.is_cuda(); 
					logging::log_always("TORCH: out ndim: ", output.ndimension(), ", ", output.sizes());

					

					auto outAccessor = output.accessor<float, 4>();
					size_t sampleOffset = startSampleValid - startSample;
					for (size_t sampleIdxLocal = 0; sampleIdxLocal < sliceSizeValid; sampleIdxLocal++)
					{
						for (size_t scanlineIdx = 0; scanlineIdx < numScanlines; scanlineIdx++)
						{
							pDataOut->get()[(startSampleValid + sampleIdxLocal) * numScanlines + scanlineIdx] = outAccessor[0][0][scanlineIdx][sampleIdxLocal + sampleOffset];
						}
					}

				}
				cudaSafeCall(cudaDeviceSynchronize());
				//std::shared_ptr<USImageProperties> newProps = make_shared<USImageProperties>(*pImageProperties);
				
				// Wrap the returned Container in an USImage with the same size etc.
				pOut = make_shared<USImage>(
					vec2s{ numScanlines, numSamples },
					pDataOut,
					pImageProperties,
					0,
					0);

				//// Wrap the returned Container in an USRawData with the same size etc.
				//pOut = make_shared<USRawData>(
				//	pInRawData->getNumScanlines(),
				//	pInRawData->getNumElements(),
				//	pInRawData->getElementLayout(),
				//	pInRawData->getNumReceivedChannels(),
				//	pInRawData->getNumSamples(),
				//	pInRawData->getSamplingFrequency(),
				//	pProcessedData,
				//	pInRawData->getRxBeamformerParameters(),
				//	pInRawData->getImageProperties(),
				//	pInRawData->getReceiveTimestamp(),
				//	pInRawData->getSyncTimestamp());
			}
			catch (c10::Error e)
			{
				logging::log_error("TorchNode: Error (c10::Error) while running model '", m_modelFilename, "'");
				logging::log_error("TorchNode: ", e.what());
				logging::log_error("TorchNode: ", e.msg_stack());
			}
			catch (std::runtime_error e)
			{
				logging::log_error("TorchNode: Error (std::runtime_error) while running model '", m_modelFilename, "'");
				logging::log_error("TorchNode: ", e.what());
			}
		//case supra::TypeUint8:
			//return m_compensator->process<InputType, uint8_t>(imageData, size, workDimension);
			//break;
		//case supra::TypeInt16:
			//return m_compensator->process<InputType, int16_t>(imageData, size, workDimension);
			//break;
		//case supra::TypeFloat:
			//return m_compensator->process<InputType, float>(imageData, size, workDimension);
			//break;
		//default:
			//logging::log_error("TorchNode: Output image type not supported");
			//break;
		}
		return pOut;
	}

	shared_ptr<RecordObject> TorchNode::checkTypeAndProcess(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<RecordObject> pOut = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
			if (pInImage)
			{
				// lock the object mutex to make sure no parameters are changed during processing
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types, 
				// only those meaningful for the operation of the node.
				switch (pInImage->getDataType())
				{
				case TypeUint8:
					pOut = processTemplateSelection<uint8_t>(pInImage->getData<uint8_t>(), pInImage->getSize(), 1, pInImage->getImageProperties());
					break;
				case TypeInt16:
					pOut = processTemplateSelection<int16_t>(pInImage->getData<int16_t>(), pInImage->getSize(), 1, pInImage->getImageProperties());
					break;
				case TypeFloat:
					pOut = processTemplateSelection<float>(pInImage->getData<float>(), pInImage->getSize(), 1, pInImage->getImageProperties());
					break;
				default:
					logging::log_error("TorchNode: Input image type not supported");
					break;
				}
				m_callFrequency.measureEnd();
			}
			else {
				logging::log_error("TorchNode: could not cast object to USImage type, although its type is TypeUSImage.");
			}
		}
		else if (inObj && inObj->getType() == TypeUSRawData)
		{
			shared_ptr<USRawData> pInRawData = dynamic_pointer_cast<USRawData>(inObj);
			if (pInRawData)
			{
				// lock the object mutex to make sure no parameters are changed during processing
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types,
				// only those meaningful for the operation of the node.
				vec3s size{pInRawData->getNumSamples(), pInRawData->getNumReceivedChannels(), pInRawData->getNumScanlines()};
				switch (pInRawData->getDataType())
				{
					case TypeUint8:
						pOut = processTemplateSelection<uint8_t>(pInRawData->getData<uint8_t>(), size, 0, pInRawData->getImageProperties());
						break;
					case TypeInt16:
						pOut = processTemplateSelection<int16_t>(pInRawData->getData<int16_t>(), size, 0, pInRawData->getImageProperties());
						break;
					case TypeFloat:
						pOut = processTemplateSelection<float>(pInRawData->getData<float>(), size, 0, pInRawData->getImageProperties());
						break;
					default:
						logging::log_error("TorchNode: Input image type not supported");
						break;
				}
				m_callFrequency.measureEnd();
			}
			else {
				logging::log_error("TorchNode: could not cast object to USRawData type, although its type is TypeUSRawData.");
			}
		}
		return pOut;
	}

	void TorchNode::loadModule() {
		m_torchModule = nullptr;
		if (m_modelFilename != "")
		{
			try {
				std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(m_modelFilename);
				module->to(torch::kCUDA);
				m_torchModule = module;
			}
			catch (c10::Error e)
			{
				logging::log_error("TorchNode: Exception (c10::Error) while loading model '", m_modelFilename, "'");
				logging::log_error("TorchNode: ", e.what());
				logging::log_error("TorchNode: ", e.msg_stack());
				m_torchModule = nullptr;
			}
			catch (std::runtime_error e)
			{
				logging::log_error("TorchNode: Exception (std::runtime_error) while loading model '", m_modelFilename, "'");
				logging::log_error("TorchNode: ", e.what());
				m_torchModule = nullptr;
			}
		}
	}
}