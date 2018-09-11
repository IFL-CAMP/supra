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

#include "IQDemodulatorNode.h"
#include "IQDemodulator.h"

#include "USImage.h"
#include <utilities/Logging.h>

//TODO remove this later
#include <utilities/cudaUtility.h>

using namespace std;

namespace supra
{
	IQDemodulatorNode::IQDemodulatorNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_samplingFrequency(40 * 1e6)
		, m_cutoffFrequency(2.5 * 1e6)
		, m_decimationLowpassFilterLength(51)
		, m_frequencyCompoundingBandpassFilterLength(51)
		, m_frequencyCompoundingReferenceFrequencies(6)
		, m_frequencyCompoundingBandwidths(6, 1e6)
		, m_frequencyCompoundingWeights(6)
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
		m_callFrequency.setName("IQDem");

		//TODO expose bandwidths
		m_valueRangeDictionary.set<uint32_t>("decimation", 1, 100, 8, "Decimation");
		m_valueRangeDictionary.set<double>("referenceFrequency", 1 * 1e6, 20 * 1e6, 7 * 1e6, "Reference frequency");
		m_valueRangeDictionary.set<double>("referenceFrequencyAdd0", 0, 20 * 1e6, 0, "Reference frequency ADD0");
		m_valueRangeDictionary.set<double>("referenceFrequencyAdd1", 0, 20 * 1e6, 0, "Reference frequency ADD1");
		m_valueRangeDictionary.set<double>("referenceFrequencyAdd2", 0, 20 * 1e6, 0, "Reference frequency ADD2");
		m_valueRangeDictionary.set<double>("referenceFrequencyAdd3", 0, 20 * 1e6, 0, "Reference frequency ADD3");
		m_valueRangeDictionary.set<double>("referenceFrequencyAdd4", 0, 20 * 1e6, 0, "Reference frequency ADD4");
		m_valueRangeDictionary.set<double>("weight", 0, 5, 1, "Weight");
		m_valueRangeDictionary.set<double>("weightAdd0", 0, 5, 0, "Weight ADD0");
		m_valueRangeDictionary.set<double>("weightAdd1", 0, 5, 0, "Weight ADD1");
		m_valueRangeDictionary.set<double>("weightAdd2", 0, 5, 0, "Weight ADD2");
		m_valueRangeDictionary.set<double>("weightAdd3", 0, 5, 0, "Weight ADD3");
		m_valueRangeDictionary.set<double>("weightAdd4", 0, 5, 0, "Weight ADD4");
		m_valueRangeDictionary.set<double>("bandwidth", 0, 20 * 1e6, 1e6, "Bandwidth");
		m_valueRangeDictionary.set<double>("bandwidthAdd0", 0, 20 * 1e6, 1e6, "Bandwidth ADD0");
		m_valueRangeDictionary.set<double>("bandwidthAdd1", 0, 20 * 1e6, 1e6, "Bandwidth ADD1");
		m_valueRangeDictionary.set<double>("bandwidthAdd2", 0, 20 * 1e6, 1e6, "Bandwidth ADD2");
		m_valueRangeDictionary.set<double>("bandwidthAdd3", 0, 20 * 1e6, 1e6, "Bandwidth ADD3");
		m_valueRangeDictionary.set<double>("bandwidthAdd4", 0, 20 * 1e6, 1e6, "Bandwidth ADD4");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeInt16 }, TypeFloat, "Output type");
		configurationChanged();

		m_demodulator = unique_ptr<IQDemodulator>(new IQDemodulator(m_samplingFrequency, m_referenceFrequency, m_cutoffFrequency, m_decimationLowpassFilterLength, m_frequencyCompoundingBandpassFilterLength));
	}

	void IQDemodulatorNode::configurationChanged()
	{
		m_decimation = m_configurationDictionary.get<uint32_t>("decimation");
		readFrequencyCompoundingSettings();
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
	}

	void IQDemodulatorNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "decimation")
		{
			m_decimation = m_configurationDictionary.get<uint32_t>("decimation");
		}
		else if (configKey == "referenceFrequency" ||
			configKey == "referenceFrequencyAdd0" ||
			configKey == "referenceFrequencyAdd1" ||
			configKey == "referenceFrequencyAdd2" ||
			configKey == "referenceFrequencyAdd3" ||
			configKey == "referenceFrequencyAdd4" ||
			configKey == "weight" ||
			configKey == "weightAdd0" ||
			configKey == "weightAdd1" ||
			configKey == "weightAdd2" ||
			configKey == "weightAdd3" ||
			configKey == "weightAdd4" ||
			configKey == "bandwidth" ||
			configKey == "bandwidthAdd0" ||
			configKey == "bandwidthAdd1" ||
			configKey == "bandwidthAdd2" ||
			configKey == "bandwidthAdd3" ||
			configKey == "bandwidthAdd4")
		{
			readFrequencyCompoundingSettings();
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

	void IQDemodulatorNode::readFrequencyCompoundingSettings()
	{
		m_referenceFrequency = m_configurationDictionary.get<double>("referenceFrequency");
		m_frequencyCompoundingReferenceFrequencies.resize(6, 0.0);
		m_frequencyCompoundingReferenceFrequencies[0] = m_referenceFrequency;
		m_frequencyCompoundingReferenceFrequencies[1] = m_configurationDictionary.get<double>("referenceFrequencyAdd0");
		m_frequencyCompoundingReferenceFrequencies[2] = m_configurationDictionary.get<double>("referenceFrequencyAdd1");
		m_frequencyCompoundingReferenceFrequencies[3] = m_configurationDictionary.get<double>("referenceFrequencyAdd2");
		m_frequencyCompoundingReferenceFrequencies[4] = m_configurationDictionary.get<double>("referenceFrequencyAdd3");
		m_frequencyCompoundingReferenceFrequencies[5] = m_configurationDictionary.get<double>("referenceFrequencyAdd4");
		m_frequencyCompoundingWeights.resize(6, 0.0);
		m_frequencyCompoundingWeights[0] = m_configurationDictionary.get<double>("weight");
		m_frequencyCompoundingWeights[1] = m_configurationDictionary.get<double>("weightAdd0");
		m_frequencyCompoundingWeights[2] = m_configurationDictionary.get<double>("weightAdd1");
		m_frequencyCompoundingWeights[3] = m_configurationDictionary.get<double>("weightAdd2");
		m_frequencyCompoundingWeights[4] = m_configurationDictionary.get<double>("weightAdd3");
		m_frequencyCompoundingWeights[5] = m_configurationDictionary.get<double>("weightAdd4");
		m_frequencyCompoundingBandwidths.resize(6, 1e6);
		m_frequencyCompoundingBandwidths[0] = m_configurationDictionary.get<double>("bandwidth");
		m_frequencyCompoundingBandwidths[1] = m_configurationDictionary.get<double>("bandwidthAdd0");
		m_frequencyCompoundingBandwidths[2] = m_configurationDictionary.get<double>("bandwidthAdd1");
		m_frequencyCompoundingBandwidths[3] = m_configurationDictionary.get<double>("bandwidthAdd2");
		m_frequencyCompoundingBandwidths[4] = m_configurationDictionary.get<double>("bandwidthAdd3");
		m_frequencyCompoundingBandwidths[5] = m_configurationDictionary.get<double>("bandwidthAdd4");

		size_t listLength = 6;
		for (size_t k = 0; k < m_frequencyCompoundingReferenceFrequencies.size(); k++)
		{
			if (m_frequencyCompoundingReferenceFrequencies[k] == 0.0 ||
				m_frequencyCompoundingWeights[k] == 0.0)
			{
				listLength = k;
				break;
			}
		}
		m_frequencyCompoundingReferenceFrequencies.resize(listLength, 0.0);
		m_frequencyCompoundingWeights.resize(listLength, 0.0);
		m_frequencyCompoundingBandwidths.resize(listLength, 1e6);
	}

	void IQDemodulatorNode::updateImageProperties(const std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		shared_ptr<USImageProperties> newProps = make_shared<USImageProperties>(*imageProperties);
		newProps->setImageState(USImageProperties::EnvDetected);
		m_resultingSize.x = m_lastSeenImageProperties->getNumScanlines();
		m_resultingSize.y = m_demodulator->decimatedSignalLength((int)m_lastSeenImageProperties->getNumSamples(), (uint32_t)m_decimation);
		m_resultingSize.z = 1;
		newProps->setNumSamples(m_resultingSize.y);

		newProps->setSpecificParameter("Iqdemod.Decimation", m_decimation);
		newProps->setSpecificParameter("Iqdemod.FrequencyCompoundingReferenceFrequencies", stringify(m_frequencyCompoundingReferenceFrequencies));
		newProps->setSpecificParameter("Iqdemod.FrequencyCompoundingWeights", stringify(m_frequencyCompoundingWeights));
		newProps->setSpecificParameter("Iqdemod.FrequencyCompoundingBandwidths", stringify(m_frequencyCompoundingBandwidths));
		newProps->setSpecificParameter("Iqdemod.SamplingFrequency", m_samplingFrequency);
		newProps->setSpecificParameter("Iqdemod.CutoffFrequency", m_cutoffFrequency);

		m_editedImageProperties = const_pointer_cast<const USImageProperties>(newProps);
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> IQDemodulatorNode::demodulateTemplated(const std::shared_ptr<USImage> inImage)
	{
		switch (m_outputType)
		{
		case TypeInt16:
			return m_demodulator->demodulateMagnitudeFrequencyCompounding<InputType, int16_t>(
				inImage->getData<InputType>(),
				static_cast<int>(inImage->getSize().x),
				static_cast<int>(inImage->getSize().y),
				m_decimation,
				m_frequencyCompoundingReferenceFrequencies,
				m_frequencyCompoundingBandwidths,
				m_frequencyCompoundingWeights);
		case TypeFloat:
			return m_demodulator->demodulateMagnitudeFrequencyCompounding<InputType, float>(
				inImage->getData<InputType>(),
				static_cast<int>(inImage->getSize().x),
				static_cast<int>(inImage->getSize().y),
				m_decimation,
				m_frequencyCompoundingReferenceFrequencies,
				m_frequencyCompoundingBandwidths,
				m_frequencyCompoundingWeights);
		default:
			logging::log_error("IQDemodulatorNode: Output image type not supported.");
		}
		return nullptr;
	}

	shared_ptr<RecordObject> IQDemodulatorNode::checkTypeAndDemodulate(const shared_ptr<RecordObject> inObj)
	{
		typedef std::chrono::high_resolution_clock Clock;
		typedef std::chrono::milliseconds milliseconds;
		Clock::time_point t0 = Clock::now();

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
					pImageDemodData = demodulateTemplated<int16_t>(pInImage);
					break;
				case TypeFloat:
					pImageDemodData = demodulateTemplated<float>(pInImage);
					break;
				default:
					logging::log_error("IQDemodulatorNode: Input image type not supported.");
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
				logging::log_error("IQDemodulatorNode: could not cast object to USImage type, is it in suppored ElementType?");
			}
		}

		Clock::time_point t1 = Clock::now();
		milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
		//std::cout << "Time in IQ demodulation: " << ms.count() << "ms\n";
		return pImageDemod;
	}
}
