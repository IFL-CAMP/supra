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

#include "LogCompressorNode.h"
#include "LogCompressor.h"

#include "USImage.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	LogCompressorNode::LogCompressorNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_dynamicRange(80)
		, m_gain(1)
		, m_inputMax(1024)
		, m_editedImageProperties(nullptr)
	{
		if (queueing)
		{
			m_node = unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndCompress(inObj); }));
		}
		else
		{
			m_node = unique_ptr<NodeTypeDiscarding>(
				new NodeTypeDiscarding(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndCompress(inObj); }));
		}

		m_compressor = unique_ptr<LogCompressor>(new LogCompressor());
		m_callFrequency.setName("Comp");
		m_valueRangeDictionary.set<double>("dynamicRange", 1, 200, 80, "Dynamic Range");
		m_valueRangeDictionary.set<double>("gain", 0.01, 2, 1, "Gain");
		m_valueRangeDictionary.set<double>("inMax", 10, 50000, 1024, "Max Input");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeUint8 }, TypeFloat, "Output type");
		configurationChanged();
	}

	void LogCompressorNode::configurationChanged()
	{
		m_dynamicRange = m_configurationDictionary.get<double>("dynamicRange");
		m_gain = m_configurationDictionary.get<double>("gain");
		m_inputMax = m_configurationDictionary.get<double>("inMax");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
	}

	void LogCompressorNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "dynamicRange")
		{
			m_dynamicRange = m_configurationDictionary.get<double>("dynamicRange");
		}
		else if (configKey == "gain")
		{
			m_gain = m_configurationDictionary.get<double>("gain");
		}
		else if (configKey == "inMax")
		{
			m_inputMax = m_configurationDictionary.get<double>("inMax");
		}
		else if (configKey == "outputType")
		{
			m_outputType = m_configurationDictionary.get<DataType>("outputType");
		}

		if (m_lastSeenImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> LogCompressorNode::compressTemplated(std::shared_ptr<const ContainerBase> imageData, vec3s size)
	{
		auto inContainer = std::dynamic_pointer_cast<const Container<InputType>>(imageData);
		switch (m_outputType)
		{
		case supra::TypeUint8:
			return m_compressor->compress<InputType, uint8_t>(inContainer, size, m_dynamicRange, m_gain, m_inputMax);
			break;
		case supra::TypeFloat:
			return m_compressor->compress<InputType, float>(inContainer, size, m_dynamicRange, m_gain, m_inputMax);
			break;
		default:
			logging::log_error("LogCompressorNode: Output image type not supported");
			break;
		}
		return nullptr;
	}

	shared_ptr<RecordObject> LogCompressorNode::checkTypeAndCompress(shared_ptr<RecordObject> inObj)
	{
		typedef std::chrono::high_resolution_clock Clock;
		typedef std::chrono::milliseconds milliseconds;
		Clock::time_point t0 = Clock::now();

		shared_ptr<USImage> pImage = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
			if (pInImage)
			{
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();

				std::shared_ptr<ContainerBase> pImageCompressedData;
				switch (pInImage->getDataType())
				{
				case TypeUint8:
					pImageCompressedData = compressTemplated<uint8_t>(pInImage->getData<uint8_t>(), pInImage->getSize());
					break;
				case TypeInt16:
					pImageCompressedData = compressTemplated<int16_t>(pInImage->getData<int16_t>(), pInImage->getSize());
					break;
				case TypeFloat:
					pImageCompressedData = compressTemplated<float>(pInImage->getData<float>(), pInImage->getSize());
					break;
				default:
					logging::log_error("LogCompressorNode: Input image type not supported");
					break;
				}
				m_callFrequency.measureEnd();				

				if (pInImage->getImageProperties() != m_lastSeenImageProperties)
				{
					updateImageProperties(pInImage->getImageProperties());
				}

				//logging::log_log("LC: ", pInImage->getSize().x, " ", pInImage->getSize().y, " ", pInImage->getSize().z);

				pImage = make_shared<USImage>(
					pInImage->getSize(),
					pImageCompressedData,
					m_editedImageProperties,
					pInImage->getReceiveTimestamp(),
					pInImage->getSyncTimestamp());
			}
			else {
				logging::log_error("LogCompressorNode: could not cast object to USImage type, is it in suppored ElementType?");
			}
		}
		Clock::time_point t1 = Clock::now();
		milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
		std::cout << "Time in Log Compression: " << ms.count() << "ms\n";
		return pImage;
	}

	void LogCompressorNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;

		shared_ptr<USImageProperties> newProps = make_shared<USImageProperties>(*imageProperties);
		newProps->setImageState(USImageProperties::PreScan);
		newProps->setSpecificParameter<double>("LogCompressor.DynamicRange", m_dynamicRange);
		newProps->setSpecificParameter<double>("LogCompressor.Gain", m_gain);
		newProps->setSpecificParameter<double>("LogCompressor.InMax", m_inputMax);

		m_editedImageProperties = const_pointer_cast<const USImageProperties>(newProps);
	}
}