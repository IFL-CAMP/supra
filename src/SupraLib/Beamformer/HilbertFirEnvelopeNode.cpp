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

#include "HilbertFirEnvelopeNode.h"
#include "HilbertFirEnvelope.h"

#include "USImage.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	HilbertFirEnvelopeNode::HilbertFirEnvelopeNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_needNewDemodulator(true)
	{
		if (queueing)
		{
			m_node = std::unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1,
					[this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndDemodulate(inObj); })
				);
		}
		else
		{
			m_node = std::unique_ptr<NodeTypeDiscarding>(
				new NodeTypeDiscarding(graph, 1,
					[this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndDemodulate(inObj); })
				);
		}
		m_callFrequency.setName("HilbertFirEnvelope");

		m_valueRangeDictionary.set<uint32_t>("filterLength", 1, 257, 65, "Filter Length");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeInt16 }, TypeFloat, "Output type");
		configurationChanged();
	}

	void HilbertFirEnvelopeNode::configurationChanged()
	{
		m_filterLength = m_configurationDictionary.get<uint32_t>("filterLength");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
	}

	void HilbertFirEnvelopeNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "filterLength")
		{
			m_filterLength = m_configurationDictionary.get<uint32_t>("filterLength");
			m_needNewDemodulator = true;
		}
		else if (configKey == "outputType")
		{
			m_outputType = m_configurationDictionary.get<DataType>("outputType");
		}

		m_lastSeenImageProperties = nullptr;
	}

	void HilbertFirEnvelopeNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		shared_ptr<USImageProperties> newProps = make_shared<USImageProperties>(*imageProperties);
		newProps->setImageState(USImageProperties::EnvDetected);
		m_resultingSize.x = m_lastSeenImageProperties->getNumScanlines();
		m_resultingSize.y = m_lastSeenImageProperties->getNumSamples();
		m_resultingSize.z = 1;
		newProps->setNumSamples(m_resultingSize.y);

		newProps->setSpecificParameter("HilbertFirEnvelope.FilterLength", m_filterLength);

		m_editedImageProperties = const_pointer_cast<const USImageProperties>(newProps);
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> HilbertFirEnvelopeNode::demodulateTemplated(std::shared_ptr<USImage> inImage)
	{
		switch (m_outputType)
		{
		case TypeInt16:
			return m_demodulator->demodulate<InputType, int16_t>(
				inImage->getData<InputType>(),
				static_cast<int>(inImage->getSize().x),
				static_cast<int>(inImage->getSize().y));
		case TypeFloat:
			return m_demodulator->demodulate<InputType, float>(
				inImage->getData<InputType>(),
				static_cast<int>(inImage->getSize().x),
				static_cast<int>(inImage->getSize().y));
		default:
			logging::log_error("HilbertFirEnvelopeNode: Output image type not supported.");
		}
		return nullptr;
	}

	shared_ptr<RecordObject> HilbertFirEnvelopeNode::checkTypeAndDemodulate(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<USImage> pImageDemod = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
			if (pInImage)
			{
				unique_lock<mutex> l(m_mutex);			
				m_callFrequency.measure();

				if (m_needNewDemodulator)
				{
					m_demodulator = unique_ptr<HilbertFirEnvelope>(new HilbertFirEnvelope(m_filterLength));
					m_lastSeenImageProperties = nullptr;
					m_needNewDemodulator = false;
				}

				std::shared_ptr<ContainerBase> pImageDemodData;
				switch (pInImage->getDataType())
				{
				case TypeInt16:
					pImageDemodData = demodulateTemplated<int16_t>(pInImage);
					break;
				case TypeFloat:
					pImageDemodData = demodulateTemplated<float>(pInImage);
					break;
				default:
					logging::log_error("HilbertFirEnvelopeNode: Input image type not supported.");
				}
				
				m_callFrequency.measureEnd();
				
				if (pInImage->getImageProperties() != m_lastSeenImageProperties)
				{
					updateImageProperties(pInImage->getImageProperties());
				}

				pImageDemod = make_shared<USImage>(
					m_resultingSize,
					pImageDemodData,
					m_editedImageProperties,
					pInImage->getReceiveTimestamp(),
					pInImage->getSyncTimestamp());
			}
			else {
				logging::log_error("HilbertFirEnvelopeNode: could not cast object to USImage type, is it in suppored ElementType?");
			}
		}
		return pImageDemod;
	}
}
