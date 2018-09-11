// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2018, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "NoiseNode.h"
#include "NoiseCuda.h"

#include "USImage.h"
#include "Beamformer/USRawData.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	NoiseNode::NoiseNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
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

		m_callFrequency.setName("NoiseNode");

		// Define the parameters that this node reveals to the user
		m_valueRangeDictionary.set<double>("additiveUniformMin", -1000.0, 1000.0, 0.0, "additiveUniformMin");
		m_valueRangeDictionary.set<double>("additiveUniformMax", -1000.0, 1000.0, 0.0, "additiveUniformMax");
		m_valueRangeDictionary.set<double>("additiveGaussMean", -1000.0, 1000.0, 0.0, "additiveGaussMean");
		m_valueRangeDictionary.set<double>("additiveGaussStd", -1000.0, 1000.0, 0.0, "additiveGaussStd");
		m_valueRangeDictionary.set<double>("multiplicativeUniformMin", -5.0, 5.0, 1.0, "multiplicativeUniformMin");
		m_valueRangeDictionary.set<double>("multiplicativeUniformMax", -5.0, 5.0, 1.0, "multiplicativeUniformMax");
		m_valueRangeDictionary.set<double>("multiplicativeGaussMean", -5.0, 5.0, 1.0, "multiplicativeGaussMean");
		m_valueRangeDictionary.set<double>("multiplicativeGaussStd", -10.0, 10.0, 0.0, "multiplicativeGaussStd");
		m_valueRangeDictionary.set<bool>("additiveUniformCorrelated", false, "additiveUniformCorrelated");
		m_valueRangeDictionary.set<bool>("additiveGaussCorrelated", false, "additiveGaussCorrelated");
		m_valueRangeDictionary.set<bool>("multiplicativeUniformCorrelated", false, "multiplicativeUniformCorrelated");
		m_valueRangeDictionary.set<bool>("multiplicativeGaussCorrelated", false, "multiplicativeGaussCorrelated");
	
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeUint8, TypeInt16 }, TypeFloat, "Output type");

		// read the configuration to apply the default values
		configurationChanged();
	}

	void NoiseNode::configurationChanged()
	{
		m_additiveUniformMin = m_configurationDictionary.get<double>("additiveUniformMin");
		m_additiveUniformMax = m_configurationDictionary.get<double>("additiveUniformMax");
		m_additiveGaussMean = m_configurationDictionary.get<double>("additiveGaussMean");
		m_additiveGaussStd = m_configurationDictionary.get<double>("additiveGaussStd");
		m_multiplicativeUniformMin = m_configurationDictionary.get<double>("multiplicativeUniformMin");
		m_multiplicativeUniformMax = m_configurationDictionary.get<double>("multiplicativeUniformMax");
		m_multiplicativeGaussMean = m_configurationDictionary.get<double>("multiplicativeGaussMean");
		m_multiplicativeGaussStd = m_configurationDictionary.get<double>("multiplicativeGaussStd");
		m_additiveUniformCorrelated = m_configurationDictionary.get<bool>("additiveUniformCorrelated");
		m_additiveGaussCorrelated = m_configurationDictionary.get<bool>("additiveGaussCorrelated");
		m_multiplicativeUniformCorrelated = m_configurationDictionary.get<bool>("multiplicativeUniformCorrelated");
		m_multiplicativeGaussCorrelated = m_configurationDictionary.get<bool>("multiplicativeGaussCorrelated");

		m_outputType = m_configurationDictionary.get<DataType>("outputType");
	}

	void NoiseNode::configurationEntryChanged(const std::string& configKey)
	{
		// lock the object mutex to make sure no processing happens during parameter changes
		unique_lock<mutex> l(m_mutex);
		if (configKey == "additiveUniformMin")
		{
			m_additiveUniformMin = m_configurationDictionary.get<double>("additiveUniformMin");
		}
		else if (configKey == "additiveUniformMax")
		{
			m_additiveUniformMax = m_configurationDictionary.get<double>("additiveUniformMax");
		}
		else if (configKey == "additiveGaussMean")
		{
			m_additiveGaussMean = m_configurationDictionary.get<double>("additiveGaussMean");
		}
		else if (configKey == "additiveGaussStd")
		{
			m_additiveGaussStd = m_configurationDictionary.get<double>("additiveGaussStd");
		}
		else if (configKey == "multiplicativeUniformMin")
		{
			m_multiplicativeUniformMin = m_configurationDictionary.get<double>("multiplicativeUniformMin");
		}
		else if (configKey == "multiplicativeUniformMax")
		{
			m_multiplicativeUniformMax = m_configurationDictionary.get<double>("multiplicativeUniformMax");
		}
		else if (configKey == "multiplicativeGaussMean")
		{
			m_multiplicativeGaussMean = m_configurationDictionary.get<double>("multiplicativeGaussMean");
		}
		else if (configKey == "multiplicativeGaussStd")
		{
			m_multiplicativeGaussStd = m_configurationDictionary.get<double>("multiplicativeGaussStd");
		}
		else if (configKey == "additiveUniformCorrelated")
		{
			m_additiveUniformCorrelated = m_configurationDictionary.get<bool>("additiveUniformCorrelated");
		}
		else if (configKey == "additiveGaussCorrelated")
		{
			m_additiveGaussCorrelated = m_configurationDictionary.get<bool>("additiveGaussCorrelated");
		}
		else if (configKey == "multiplicativeUniformCorrelated")
		{
			m_multiplicativeUniformCorrelated = m_configurationDictionary.get<bool>("multiplicativeUniformCorrelated");
		}
		else if (configKey == "multiplicativeGaussCorrelated")
		{
			m_multiplicativeGaussCorrelated = m_configurationDictionary.get<bool>("multiplicativeGaussCorrelated");
		}		
		else if (configKey == "outputType")
		{
			m_outputType = m_configurationDictionary.get<DataType>("outputType");
		}
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> NoiseNode::processTemplateSelection(std::shared_ptr<const Container<InputType> > imageData, vec3s size)
	{
		// With the function already templated on the input type, handle the desired output type.
		switch (m_outputType)
		{
		case supra::TypeUint8:
			return NoiseCuda::process<InputType, uint8_t>(imageData, size, 
				static_cast<NoiseCuda::WorkType>(m_additiveUniformMin),
				static_cast<NoiseCuda::WorkType>(m_additiveUniformMax),
				static_cast<NoiseCuda::WorkType>(m_additiveGaussMean),
				static_cast<NoiseCuda::WorkType>(m_additiveGaussStd),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeUniformMin),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeUniformMax),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeGaussMean),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeGaussStd),
				m_additiveUniformCorrelated,
				m_additiveGaussCorrelated,
				m_multiplicativeUniformCorrelated,
				m_multiplicativeGaussCorrelated);
			break;
		case supra::TypeInt16:
			return NoiseCuda::process<InputType, int16_t>(imageData, size,
				static_cast<NoiseCuda::WorkType>(m_additiveUniformMin),
				static_cast<NoiseCuda::WorkType>(m_additiveUniformMax),
				static_cast<NoiseCuda::WorkType>(m_additiveGaussMean),
				static_cast<NoiseCuda::WorkType>(m_additiveGaussStd),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeUniformMin),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeUniformMax),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeGaussMean),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeGaussStd),
				m_additiveUniformCorrelated,
				m_additiveGaussCorrelated,
				m_multiplicativeUniformCorrelated,
				m_multiplicativeGaussCorrelated);
			break;
		case supra::TypeFloat:
			return NoiseCuda::process<InputType, float>(imageData, size,
				static_cast<NoiseCuda::WorkType>(m_additiveUniformMin),
				static_cast<NoiseCuda::WorkType>(m_additiveUniformMax),
				static_cast<NoiseCuda::WorkType>(m_additiveGaussMean),
				static_cast<NoiseCuda::WorkType>(m_additiveGaussStd),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeUniformMin),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeUniformMax),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeGaussMean),
				static_cast<NoiseCuda::WorkType>(m_multiplicativeGaussStd),
				m_additiveUniformCorrelated,
				m_additiveGaussCorrelated,
				m_multiplicativeUniformCorrelated,
				m_multiplicativeGaussCorrelated);
			break;
		default:
			logging::log_error("NoiseNode: Output image type not supported");
			break;
		}
		return nullptr;
	}

	shared_ptr<RecordObject> NoiseNode::checkTypeAndProcess(shared_ptr<RecordObject> inObj)
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

				std::shared_ptr<ContainerBase> pImageProcessedData;

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types, 
				// only those meaningful for the operation of the node.
				switch (pInImage->getDataType())
				{
				case TypeUint8:
					pImageProcessedData = processTemplateSelection<uint8_t>(pInImage->getData<uint8_t>(), pInImage->getSize());
					break;
				case TypeInt16:
					pImageProcessedData = processTemplateSelection<int16_t>(pInImage->getData<int16_t>(), pInImage->getSize());
					break;
				case TypeFloat:
					pImageProcessedData = processTemplateSelection<float>(pInImage->getData<float>(), pInImage->getSize());
					break;
				default:
					logging::log_error("NoiseNode: Input image type not supported");
					break;
				}
				m_callFrequency.measureEnd();

				// Wrap the returned Container in an USImage with the same size etc.
				pOut = make_shared<USImage>(
					pInImage->getSize(),
					pImageProcessedData,
					pInImage->getImageProperties(),
					pInImage->getReceiveTimestamp(),
					pInImage->getSyncTimestamp());
			}
			else {
				logging::log_error("NoiseNode: could not cast object to USImage type, is it in suppored ElementType?");
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

				std::shared_ptr<ContainerBase> pImageProcessedData;

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types, 
				// only those meaningful for the operation of the node.
				vec3s size{ pInRawData->getNumReceivedChannels(), pInRawData->getNumSamples() , pInRawData->getNumScanlines() };
				switch (pInRawData->getDataType())
				{
				case TypeUint8:
					pImageProcessedData = processTemplateSelection<uint8_t>(pInRawData->getData<uint8_t>(), size);
					break;
				case TypeInt16:
					pImageProcessedData = processTemplateSelection<int16_t>(pInRawData->getData<int16_t>(), size);
					break;
				case TypeFloat:
					pImageProcessedData = processTemplateSelection<float>(pInRawData->getData<float>(), size);
					break;
				default:
					logging::log_error("NoiseNode: Input image type not supported");
					break;
				}
				m_callFrequency.measureEnd();

				// Wrap the returned Container in an USRawData with the same size etc.
				pOut = std::make_shared<USRawData>
					(pInRawData->getNumScanlines(),
						pInRawData->getNumElements(),
						pInRawData->getElementLayout(),
						pInRawData->getNumReceivedChannels(),
						pInRawData->getNumSamples(),
						pInRawData->getSamplingFrequency(),
						pImageProcessedData,
						pInRawData->getRxBeamformerParameters(),
						pInRawData->getImageProperties(),
						pInRawData->getReceiveTimestamp(),
						pInRawData->getSyncTimestamp());
			}
			else {
				logging::log_error("NoiseNode: could not cast object to USRawData type, is it in suppored ElementType?");
			}
		}
		return pOut;
	}
}