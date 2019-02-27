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
#include "TorchInference.h"

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
		m_valueRangeDictionary.set<DataType>("nodeOutputDataType", { TypeInt8, TypeUint8, TypeInt16, TypeUint16, TypeInt32, TypeInt64, TypeFloat, TypeDouble }, TypeFloat, "Node output type");
		m_valueRangeDictionary.set<string>("modelFilename", "model.pt", "Model filename");
		m_valueRangeDictionary.set<DataType>("modelInputDataType", { TypeInt8, TypeUint8, TypeInt16, TypeUint16, TypeInt32, TypeInt64, TypeFloat, TypeDouble }, TypeFloat, "Model input datatype");
		m_valueRangeDictionary.set<DataType>("modelOutputDataType", { TypeInt8, TypeUint8, TypeInt16, TypeUint16, TypeInt32, TypeInt64, TypeFloat, TypeDouble }, TypeFloat, "Model input datatype");
		m_valueRangeDictionary.set<string>("modelInputClass", { "USRawData", "USImage" }, "USImage", "Model input class");
		m_valueRangeDictionary.set<string>("modelOutputClass", { "USRawData", "USImage" }, "USImage", "Model output class");
		m_valueRangeDictionary.set<string>("modelInputLayout", { "CxWxH", "WxHxC", "CxHxW", "HxWxC", "WxCxH", "HxCxW" }, "CxWxH", "Model input layout");
		m_valueRangeDictionary.set<string>("modelOutputLayout", { "CxWxH", "WxHxC", "CxHxW", "HxWxC", "WxCxH", "HxCxW" }, "CxWxH", "Model input layout");
		m_valueRangeDictionary.set<uint32_t>("inferencePatchSize", 0, "Inference patch size");
		m_valueRangeDictionary.set<uint32_t>("inferencePatchOverlap", 0, "Inference patch overlap");
		
		// read the configuration to apply the default values
		configurationChanged();
	}

	void TorchNode::configurationChanged()
	{
		m_nodeOutputDataType =  m_configurationDictionary.get<DataType>("nodeOutputDataType");
		m_modelFilename =       m_configurationDictionary.get<string>("modelFilename");
		m_modelInputDataType =  m_configurationDictionary.get<DataType>("modelInputDataType");
		m_modelOutputDataType = m_configurationDictionary.get<DataType>("modelOutputDataType");
		m_modelInputClass =     m_configurationDictionary.get<string>("modelInputClass");
		m_modelOutputClass =    m_configurationDictionary.get<string>("modelOutputClass");
		m_modelInputLayout =    m_configurationDictionary.get<string>("modelInputLayout");
		m_modelOutputLayout =   m_configurationDictionary.get<string>("modelOutputLayout");
		m_inferencePatchSize =  m_configurationDictionary.get<uint32_t>("inferencePatchSize");
		m_inferencePatchOverlap = m_configurationDictionary.get<uint32_t>("inferencePatchOverlap");

		loadModule();
	}

	void TorchNode::configurationEntryChanged(const std::string& configKey)
	{
		// lock the object mutex to make sure no processing happens during parameter changes
		unique_lock<mutex> l(m_mutex);
		if (configKey == "nodeOutputDataType")
		{
			m_nodeOutputDataType = m_configurationDictionary.get<DataType>("nodeOutputDataType");
		}
		else if (configKey == "modelFilename")
		{
			m_modelFilename = m_configurationDictionary.get<string>("modelFilename");
			loadModule();
		}
		else if (configKey == "modelInputDataType")
		{
			m_modelInputDataType = m_configurationDictionary.get<DataType>("modelInputDataType");
		}
		else if (configKey == "modelOutputDataType")
		{
			m_modelOutputDataType = m_configurationDictionary.get<DataType>("modelOutputDataType");
		}
		else if (configKey == "modelInputClass")
		{
			m_modelInputClass = m_configurationDictionary.get<string>("modelInputClass");
		}
		else if (configKey == "modelOutputClass")
		{
			m_modelOutputClass = m_configurationDictionary.get<string>("modelOutputClass");
		}
		else if (configKey == "modelInputLayout")
		{
			m_modelInputLayout = m_configurationDictionary.get<string>("modelInputLayout");
		}
		else if (configKey == "modelOutputLayout")
		{
			m_modelOutputLayout = m_configurationDictionary.get<string>("modelOutputLayout");
		}
		else if (configKey == "inferencePatchSize")
		{
			m_inferencePatchSize = m_configurationDictionary.get<uint32_t>("inferencePatchSize");
		}
		else if (configKey == "inferencePatchOverlap")
		{
			m_inferencePatchOverlap = m_configurationDictionary.get<uint32_t>("inferencePatchOverlap");
		}
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> TorchNode::processTemplateSelection(
			std::shared_ptr<const Container<InputType> > imageData,
			vec3s inputSize,
			vec3s outputSize, 
			const std::string& currentLayout,
		    const std::string& finalLayout)
	{
		// With the function already templated on the input type, handle the desired output type.
		shared_ptr<ContainerBase> pOut = nullptr;

		switch (m_nodeOutputDataType)
		{
		case supra::TypeUint8:
			return m_torchModule->process<InputType, uint8_t>(
				imageData,
				inputSize, outputSize,
				currentLayout, finalLayout,
				m_modelInputDataType, m_modelOutputDataType,
				m_modelInputLayout, m_modelOutputLayout,
				m_inferencePatchSize, m_inferencePatchOverlap);
			break;
		case supra::TypeInt16:
			return m_torchModule->process<InputType, int16_t>(
				imageData,
				inputSize, outputSize,
				currentLayout, finalLayout,
				m_modelInputDataType, m_modelOutputDataType,
				m_modelInputLayout, m_modelOutputLayout,
				m_inferencePatchSize, m_inferencePatchOverlap);
			break;
		case supra::TypeFloat:
			return m_torchModule->process<InputType, float>(
				imageData,
				inputSize, outputSize,
				currentLayout, finalLayout,
				m_modelInputDataType, m_modelOutputDataType,
				m_modelInputLayout, m_modelOutputLayout,
				m_inferencePatchSize, m_inferencePatchOverlap);
			break;
		default:
			logging::log_error("TorchNode: Output image type not supported");
			break;
		}
		return pOut;
	}

	shared_ptr<RecordObject> TorchNode::checkTypeAndProcess(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<RecordObject> pOut = nullptr;
		if (inObj)
		{
			// lock the object mutex to make sure no parameters are changed during processing
			unique_lock<mutex> l(m_mutex);

			if ((m_modelInputClass == "USImage" && inObj->getType() != TypeUSImage) || 
				(m_modelInputClass == "USRawData" && inObj->getType() != TypeUSRawData))
			{
				logging::log_error("TorchNode: Model input class is configured as '", m_modelInputClass, "', but the recieved object is type '", inObj->getType(), "'");
			}
			else if (inObj->getType() == TypeUSImage)
			{
				shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
				if (pInImage)
				{
					m_callFrequency.measure();

					// determine the output size and layout
					// Wrap the data in the output class
					vec3s inputSize = pInImage->getSize();
					vec3s outputSize;
					std::string currentLayout = "CxHxW";
					std::string finalLayout;
					if (m_modelOutputClass == "USImage")
					{
						outputSize = inputSize;
						finalLayout = currentLayout;
					}
					else if (m_modelOutputClass == "USRawData")
					{
						logging::log_warn("TorchNode: USImage->USRawData is not fully defined yet.");
						outputSize = { inputSize.y, 32, inputSize.x };
						finalLayout = "WxCxH";
					}

					std::shared_ptr<ContainerBase> pOutData;
					// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
					// This first switch handles the different input data types. There is no need to support all types, 
					// only those meaningful for the operation of the node.
					switch (pInImage->getDataType())
					{
					case TypeUint8:
						pOutData = processTemplateSelection<uint8_t>(pInImage->getData<uint8_t>(), inputSize, outputSize, currentLayout, finalLayout);
						break;
					case TypeInt16:
						pOutData = processTemplateSelection<int16_t>(pInImage->getData<int16_t>(), inputSize, outputSize, currentLayout, finalLayout);
						break;
					case TypeFloat:
						pOutData = processTemplateSelection<float>(pInImage->getData<float>(), inputSize, outputSize, currentLayout, finalLayout);
						break;
					default:
						logging::log_error("TorchNode: Input image type not supported");
						break;
					}

					// Wrap the data in the output class
					if (m_modelOutputClass == "USImage")
					{
						pOut = make_shared<USImage>(
							outputSize,
							pOutData,
							pInImage->getImageProperties(),
							pInImage->getReceiveTimestamp(),
							pInImage->getSyncTimestamp());
					}
					else if (m_modelOutputClass == "USRawData")
					{
						pOut = std::make_shared<USRawData>
							(outputSize.z,
							(size_t)128,
							vec2s{ 128, 1 },
							outputSize.y,
							outputSize.x,
							pInImage->getImageProperties()->getDepth(),
							pOutData,
							nullptr,
							pInImage->getImageProperties(),
							pInImage->getReceiveTimestamp(),
							pInImage->getSyncTimestamp());
					}
					m_callFrequency.measureEnd();
				}
				else {
					logging::log_error("TorchNode: could not cast object to USImage type, although its type is TypeUSImage.");
				}
			}
			else if (inObj->getType() == TypeUSRawData)
			{
				shared_ptr<USRawData> pInRawData = dynamic_pointer_cast<USRawData>(inObj);
				if (pInRawData)
				{
					m_callFrequency.measure();

					// determine the output size and layout
					// Wrap the data in the output class
					vec3s inputSize{ pInRawData->getNumSamples(), pInRawData->getNumReceivedChannels(), pInRawData->getNumScanlines() };
					vec3s outputSize;
					std::string currentLayout = "WxCxH";
					std::string finalLayout;
					if (m_modelOutputClass == "USImage")
					{
						outputSize = vec3s{ inputSize.z, inputSize.x, 1 };
						finalLayout = "CxHxW";
					}
					else if (m_modelOutputClass == "USRawData")
					{
						outputSize = inputSize;
						finalLayout = "WxCxH";
					}

					std::shared_ptr<ContainerBase> pOutData;
					// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
					// This first switch handles the different input data types. There is no need to support all types, 
					// only those meaningful for the operation of the node.
					switch (pInRawData->getDataType())
					{
					case TypeUint8:
						pOutData = processTemplateSelection<uint8_t>(pInRawData->getData<uint8_t>(), inputSize, outputSize, currentLayout, finalLayout);
						break;
					case TypeInt16:
						pOutData = processTemplateSelection<int16_t>(pInRawData->getData<int16_t>(), inputSize, outputSize, currentLayout, finalLayout);
						break;
					case TypeFloat:
						pOutData = processTemplateSelection<float>(pInRawData->getData<float>(), inputSize, outputSize, currentLayout, finalLayout);
						break;
					default:
						logging::log_error("TorchNode: Input image type not supported");
						break;
					}

					// Wrap the data in the output class
					if (m_modelOutputClass == "USImage")
					{
						pOut = make_shared<USImage>(
							outputSize,
							pOutData,
							pInRawData->getImageProperties(),
							pInRawData->getReceiveTimestamp(),
							pInRawData->getSyncTimestamp());
					}
					else if (m_modelOutputClass == "USRawData")
					{
						shared_ptr<USRawData> rawDataDelayed = std::make_shared<USRawData>
							(pInRawData->getNumScanlines(),
								pInRawData->getNumElements(),
								pInRawData->getElementLayout(),
								pInRawData->getNumReceivedChannels(),
								pInRawData->getNumSamples(),
								pInRawData->getSamplingFrequency(),
								pOutData,
								pInRawData->getRxBeamformerParameters(),
								pInRawData->getImageProperties(),
								pInRawData->getReceiveTimestamp(),
								pInRawData->getSyncTimestamp());
					}
					m_callFrequency.measureEnd();
				}
				else {
					logging::log_error("TorchNode: could not cast object to USRawData type, although its type is TypeUSRawData.");
				}
			}
		}
		else {
			logging::log_error("TorchNode: Input was null.");
		}
		return pOut;
	}

	void TorchNode::loadModule() {
		m_torchModule = nullptr;
		if (m_modelFilename != "")
		{
			try {
				std::shared_ptr<TorchInference> module = std::make_shared<TorchInference>(m_modelFilename);
				m_torchModule = module;
			}
			catch (std::runtime_error e)
			{
				logging::log_error("TorchNode: Exception (std::runtime_error) while creating TorchInference object from model '", m_modelFilename, "'");
				logging::log_error("TorchNode: ", e.what());
				m_torchModule = nullptr;
			}
			catch (std::exception e)
			{
				logging::log_error("TorchNode: Exception (std::exception) while creating TorchInference object from model '", m_modelFilename, "'");
				logging::log_error("TorchNode: ", e.what());
				m_torchModule = nullptr;
			}
		}
	}
}