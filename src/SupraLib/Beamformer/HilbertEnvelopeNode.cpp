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

#include "HilbertEnvelopeNode.h"
#include "HilbertEnvelope.h"

#include "USImage.h"
#include <utilities/Logging.h>

//TODO remove this later
#include <utilities/cudaUtility.h>

using namespace std;

namespace supra
{
	HilbertEnvelopeNode::HilbertEnvelopeNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
	{
		if (queueing)
		{
			m_node = std::unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1,
					[this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndEnvdetect(inObj); })
				);
		}
		else
		{
			m_node = std::unique_ptr<NodeTypeDiscarding>(
				new NodeTypeDiscarding(graph, 1,
					[this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndEnvdetect(inObj); })
				);
		}
		m_callFrequency.setName("Hilbert");

		m_valueRangeDictionary.set<uint32_t>("decimation", 1, 100, 8, "Decimation");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeInt16 }, TypeFloat, "Output type");
		configurationChanged();

		m_hilbertTransformer = unique_ptr<HilbertEnvelope>(new HilbertEnvelope());
	}

	void HilbertEnvelopeNode::configurationChanged()
	{
		m_decimation = m_configurationDictionary.get<uint32_t>("decimation");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
	}

	void HilbertEnvelopeNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "decimation")
		{
			m_decimation = m_configurationDictionary.get<uint32_t>("decimation");
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

	void HilbertEnvelopeNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		shared_ptr<USImageProperties> newProps = make_shared<USImageProperties>(*imageProperties);
		newProps->setImageState(USImageProperties::EnvDetected);
		m_resultingSize.x = m_lastSeenImageProperties->getNumScanlines();
		m_resultingSize.y = m_hilbertTransformer->decimatedSignalLength((int)m_lastSeenImageProperties->getNumSamples(), (uint32_t)m_decimation);
		m_resultingSize.z = 1;
		newProps->setNumSamples(m_resultingSize.y);

		newProps->setSpecificParameter("Hilbert.Decimation", m_decimation);

		m_editedImageProperties = const_pointer_cast<const USImageProperties>(newProps);
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> HilbertEnvelopeNode::envdetectTemplated(std::shared_ptr<USImage> inImage)
	{
		switch (m_outputType)
		{
		case TypeInt16:
			return m_hilbertTransformer->computeHilbertEnvelope<InputType, int16_t>(
				inImage->getData<InputType>(),
				static_cast<int>(inImage->getSize().x),
				static_cast<int>(inImage->getSize().y),
				m_decimation);
		case TypeFloat:
			return m_hilbertTransformer->computeHilbertEnvelope<InputType, float>(
				inImage->getData<InputType>(),
				static_cast<int>(inImage->getSize().x),
				static_cast<int>(inImage->getSize().y),
				m_decimation);
		default:
			logging::log_error("HilbertEnvelopeNode: Output image type not supported.");
		}
		return nullptr;
	}

	shared_ptr<RecordObject> HilbertEnvelopeNode::checkTypeAndEnvdetect(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<USImage> pImageDemod = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
			if (pInImage)
			{
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();
				std::shared_ptr<ContainerBase> pImageDemodData;
				switch (pInImage->getDataType())
				{
				case TypeInt16:
					pImageDemodData = envdetectTemplated<int16_t>(pInImage);
					break;
				case TypeFloat:
					pImageDemodData = envdetectTemplated<float>(pInImage);
					break;
				default:
					logging::log_error("HilbertEnvelopeNode: Input image type not supported.");
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
				logging::log_error("HilbertEnvelopeNode: could not cast object to USImage type, is it in suppored ElementType?");
			}
		}
		return pImageDemod;
	}
}
