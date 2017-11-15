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
	LogCompressorNode::LogCompressorNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_node(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndCompress(inObj); })
		, m_dynamicRange(80)
		, m_gain(1)
		, m_inputMax(1024)
		, m_editedImageProperties(nullptr)
	{
		m_compressor = unique_ptr<LogCompressor>(new LogCompressor());
		m_callFrequency.setName("Comp");
		m_valueRangeDictionary.set<double>("dynamicRange", 1, 200, 80, "Dynamic Range");
		m_valueRangeDictionary.set<double>("gain", 0.01, 2, 1, "Gain");
		m_valueRangeDictionary.set<double>("inMax", 10, 50000, 1024, "Max Input");
		configurationChanged();
	}

	void LogCompressorNode::configurationChanged()
	{
		m_dynamicRange = m_configurationDictionary.get<double>("dynamicRange");
		m_gain = m_configurationDictionary.get<double>("gain");
		m_inputMax = m_configurationDictionary.get<double>("inMax");
	}

	void LogCompressorNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "dynamicRange")
		{
			m_dynamicRange = m_configurationDictionary.get<double>("dynamicRange");
		}
		if (configKey == "gain")
		{
			m_gain = m_configurationDictionary.get<double>("gain");
		}
		if (configKey == "inMax")
		{
			m_inputMax = m_configurationDictionary.get<double>("inMax");
		}

		if (m_lastSeenImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}
	shared_ptr<RecordObject> LogCompressorNode::checkTypeAndCompress(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<USImage<uint8_t> > pImage = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage<int16_t> > pInImage = dynamic_pointer_cast<USImage<int16_t>>(inObj);
			if (pInImage)
			{
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();
				auto pImageCompressedData = m_compressor->compress<int16_t, uint8_t>(pInImage->getData(), pInImage->getSize(), m_dynamicRange, m_gain, m_inputMax);
				m_callFrequency.measureEnd();

				if (pInImage->getImageProperties() != m_lastSeenImageProperties)
				{
					updateImageProperties(pInImage->getImageProperties());
				}

				pImage = make_shared<USImage<uint8_t> >(
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