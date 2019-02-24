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
	std::shared_ptr<ContainerBase> TorchNode::processTemplateSelection(
	        std::shared_ptr<const Container<InputType> > imageData, vec3s size, size_t workDimension)
	{
		// With the function already templated on the input type, handle the desired output type.
		//switch (m_outputType)
		{
			try {
				auto inputData = torch::ones({ 1, 64, 256, 1600 });
				inputData = inputData.to(torch::kCUDA);
				inputData.set_requires_grad(false);
				assert(!(inputData.requires_grad()));

				std::vector<torch::jit::IValue> inputs;
				inputs.push_back(inputData);

				// Execute the model and turn its output into a tensor.
				auto result = m_torchModule->forward(inputs);
				at::Tensor output = result.toTensor();
				std::cout << "TORCH: " << output.ndimension() << '\n';
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
		return nullptr;
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

				std::shared_ptr<ContainerBase> pProcessedData;

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types, 
				// only those meaningful for the operation of the node.
				switch (pInImage->getDataType())
				{
				case TypeUint8:
					pProcessedData = processTemplateSelection<uint8_t>(pInImage->getData<uint8_t>(), pInImage->getSize(), 1);
					break;
				case TypeInt16:
					pProcessedData = processTemplateSelection<int16_t>(pInImage->getData<int16_t>(), pInImage->getSize(), 1);
					break;
				case TypeFloat:
					pProcessedData = processTemplateSelection<float>(pInImage->getData<float>(), pInImage->getSize(), 1);
					break;
				default:
					logging::log_error("TorchNode: Input image type not supported");
					break;
				}
				m_callFrequency.measureEnd();

				// Wrap the returned Container in an USImage with the same size etc.
				pOut = make_shared<USImage>(
					pInImage->getSize(),
					pProcessedData,
					pInImage->getImageProperties(),
					pInImage->getReceiveTimestamp(),
					pInImage->getSyncTimestamp());
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

				std::shared_ptr<ContainerBase> pProcessedData;

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types,
				// only those meaningful for the operation of the node.
				vec3s size{pInRawData->getNumSamples(), pInRawData->getNumReceivedChannels(), pInRawData->getNumScanlines()};
				switch (pInRawData->getDataType())
				{
					case TypeUint8:
						pProcessedData = processTemplateSelection<uint8_t>(pInRawData->getData<uint8_t>(), size, 0);
						break;
					case TypeInt16:
						pProcessedData = processTemplateSelection<int16_t>(pInRawData->getData<int16_t>(), size, 0);
						break;
					case TypeFloat:
						pProcessedData = processTemplateSelection<float>(pInRawData->getData<float>(), size, 0);
						break;
					default:
						logging::log_error("TorchNode: Input image type not supported");
						break;
				}
				m_callFrequency.measureEnd();

				// Wrap the returned Container in an USRawData with the same size etc.
				pOut = make_shared<USRawData>(
						pInRawData->getNumScanlines(),
						pInRawData->getNumElements(),
						pInRawData->getElementLayout(),
						pInRawData->getNumReceivedChannels(),
						pInRawData->getNumSamples(),
						pInRawData->getSamplingFrequency(),
						pProcessedData,
						pInRawData->getRxBeamformerParameters(),
						pInRawData->getImageProperties(),
						pInRawData->getReceiveTimestamp(),
						pInRawData->getSyncTimestamp());
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