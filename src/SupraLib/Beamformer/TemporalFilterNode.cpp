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

#include "TemporalFilterNode.h"
#include "TemporalFilter.h"

#include "USImage.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	TemporalFilterNode::TemporalFilterNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_editedImageProperties(nullptr)
		, m_imageSize({ 0,0,0 })
		, m_imageDataType(TypeUnknown)
	{
		if (queueing)
		{
			m_node = unique_ptr<NodeTypeQueueing>(new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return filter(inObj); }));
		}
		else
		{
			m_node = unique_ptr<NodeTypeDiscarding>(new NodeTypeDiscarding(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return filter(inObj); }));
		}

		m_temporalFilter = unique_ptr<TemporalFilter>(new TemporalFilter());
		m_callFrequency.setName("Filter");
		m_valueRangeDictionary.set<uint32_t>("numImages", 1, 50, 3, "Number of Images");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeUint16 }, TypeFloat, "Output type");
		configurationChanged();
	}

	void TemporalFilterNode::configurationChanged()
	{
		m_numImages = m_configurationDictionary.get<uint32_t>("numImages");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
	}

	void TemporalFilterNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "numImages")
		{
			m_numImages = m_configurationDictionary.get<uint32_t>("numImages");
		}
		else if (configKey == "outputType")
		{
			m_outputType = m_configurationDictionary.get<DataType>("outputType");
		}
		if (m_editedImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> TemporalFilterNode::filterTemplated(
		const std::queue<std::shared_ptr<const ContainerBase> > & inImageData,
		vec3s size,
		const std::vector<double> weights)
	{
		switch (m_outputType)
		{
		case supra::TypeInt16:
			return m_temporalFilter->filter<InputType, int16_t>(m_storedImages, m_imageSize, weights);
			break;
		case supra::TypeFloat:
			return m_temporalFilter->filter<InputType, float>(m_storedImages, m_imageSize, weights);
			break;
		default:
			logging::log_error("TemporalFilterNode: Image output type not supported");
			break;
		}
		return nullptr;
	}

	shared_ptr<RecordObject> TemporalFilterNode::filter(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<USImage> pImage = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
			if (pInImage)
			{
				unique_lock<mutex> l(m_mutex);

				if (pInImage->getSize() != m_imageSize || pInImage->getDataType() != m_imageDataType)
				{
					m_imageSize = pInImage->getSize();
					m_imageDataType = pInImage->getDataType();
					m_storedImages = decltype(m_storedImages)();
				}

				switch (m_imageDataType)
				{
				case TypeInt16:
					m_storedImages.push(pInImage->getData<int16_t>());
					break;
				case TypeFloat:
					m_storedImages.push(pInImage->getData<float>());
					break;
				default:
					break;
				}
				
				while (m_storedImages.size() > m_numImages)
				{
					m_storedImages.pop();
				}

				//compute weights:
				vector<double> weights(m_storedImages.size(), 1);

				m_callFrequency.measure();
				std::shared_ptr<ContainerBase> pImageFiltered;
				switch (m_imageDataType)
				{
				case TypeInt16:
					pImageFiltered = filterTemplated<int16_t>(m_storedImages, m_imageSize, weights);
					break;
				case TypeFloat:
					pImageFiltered = filterTemplated<float>(m_storedImages, m_imageSize, weights);
					break;
				default:
					logging::log_error("TemporalFilterNode: Image input type not supported");
					break;
				}
				m_callFrequency.measureEnd();

				if (pInImage->getImageProperties() != m_lastSeenImageProperties)
				{
					updateImageProperties(pInImage->getImageProperties());
				}

				pImage = make_shared<USImage>(
					pInImage->getSize(),
					pImageFiltered,
					m_editedImageProperties,
					pInImage->getReceiveTimestamp(),
					pInImage->getSyncTimestamp());
			}
			else {
				logging::log_error("TemporalFilterNode: could not cast object to USImage type");
			}
		}
		return pImage;
	}

	void TemporalFilterNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		shared_ptr<USImageProperties> newProps = make_shared<USImageProperties>(*imageProperties);
		newProps->setSpecificParameter<uint32_t>("TemporalFilter.NumImages", m_numImages);
		m_editedImageProperties = const_pointer_cast<const USImageProperties>(newProps);
	}
}
