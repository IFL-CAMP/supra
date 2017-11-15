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
	TemporalFilterNode::TemporalFilterNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_node(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return filter(inObj); })
		, m_editedImageProperties(nullptr)
		, m_imageSize({ 0,0,0 })
	{
		m_temporalFilter = unique_ptr<TemporalFilter>(new TemporalFilter());
		m_callFrequency.setName("Filter");
		m_valueRangeDictionary.set<uint32_t>("numImages", 1, 50, 3, "Number of Images");
		configurationChanged();
	}

	void TemporalFilterNode::configurationChanged()
	{
		m_numImages = m_configurationDictionary.get<uint32_t>("numImages");
	}

	void TemporalFilterNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "numImages")
		{
			m_numImages = m_configurationDictionary.get<uint32_t>("numImages");
		}
		if (m_editedImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}
	shared_ptr<RecordObject> TemporalFilterNode::filter(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<USImage<int16_t> > pImage = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage<int16_t> > pInImage = dynamic_pointer_cast<USImage<int16_t>>(inObj);
			if (pInImage)
			{
				unique_lock<mutex> l(m_mutex);

				if (pInImage->getSize() != m_imageSize)
				{
					m_imageSize = pInImage->getSize();
					m_storedImages = decltype(m_storedImages)();
				}

				m_storedImages.push(pInImage->getData());
				while (m_storedImages.size() > m_numImages)
				{
					m_storedImages.pop();
				}

				//compute weights:
				vector<double> weights(m_storedImages.size(), 1);

				m_callFrequency.measure();
				auto pImageFiltered = m_temporalFilter->filter<int16_t, int16_t>(m_storedImages, m_imageSize, weights);
				m_callFrequency.measureEnd();

				if (pInImage->getImageProperties() != m_lastSeenImageProperties)
				{
					updateImageProperties(pInImage->getImageProperties());
				}

				pImage = make_shared<USImage<int16_t> >(
					pInImage->getSize(),
					pImageFiltered,
					m_editedImageProperties,
					pInImage->getReceiveTimestamp(),
					pInImage->getSyncTimestamp());
			}
			else {
				logging::log_error("TemporalFilterNode: could not cast object to USImage type, is it in supported ElementType?");
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
