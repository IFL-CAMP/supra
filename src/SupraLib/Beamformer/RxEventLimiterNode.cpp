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

#include "RxEventLimiterNode.h"

#include "USRawData.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	RxEventLimiterNode::RxEventLimiterNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_lastSeenImageProperties(nullptr)
		, m_modifiedImageProperties(nullptr)
		, m_parametersRequireUpdate(true)
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

		m_callFrequency.setName("RxEventLimiterNode");

		// Define the parameters that this node reveals to the user
		m_valueRangeDictionary.set<uint32_t>("firstEventIdxToKeep", 0, 511, 0, "Index of first event to keep");
		m_valueRangeDictionary.set<uint32_t>("lastEventIdxToKeep", 0, 511, 511, "Index of last event to keep");

		// read the configuration to apply the default values
		configurationChanged();
	}

	void RxEventLimiterNode::configurationChanged()
	{
		m_firstEventIdxToKeep = m_configurationDictionary.get<uint32_t>("firstEventIdxToKeep");
		m_lastEventIdxToKeep = m_configurationDictionary.get<uint32_t>("lastEventIdxToKeep");
	}

	void RxEventLimiterNode::configurationEntryChanged(const std::string& configKey)
	{
		// lock the object mutex to make sure no processing happens during parameter changes
		unique_lock<mutex> l(m_mutex);
		if (configKey == "firstEventIdxToKeep")
		{
			m_firstEventIdxToKeep = m_configurationDictionary.get<uint32_t>("firstEventIdxToKeep");
			m_parametersRequireUpdate = true;
		}
		else if (configKey == "lastEventIdxToKeep")
		{
			m_lastEventIdxToKeep = m_configurationDictionary.get<uint32_t>("lastEventIdxToKeep");
			m_parametersRequireUpdate = true;
		}
		logging::log_error_if(m_lastEventIdxToKeep < m_firstEventIdxToKeep, 
			"RxEventLimiterNode: lastEventIdxToKeep < firstEventIdxToKeep (", 
			m_lastEventIdxToKeep, " < ", m_firstEventIdxToKeep, ")");
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> RxEventLimiterNode::processTemplateSelection(std::shared_ptr<const USRawData> inputData)
	{
		size_t numElementsPerEvent = inputData->getNumSamples()*inputData->getNumReceivedChannels();
		const InputType* ptrStart = inputData->getData<InputType>()->get() + numElementsPerEvent*m_firstEventIdxToKeep;
		const InputType* ptrEnd = inputData->getData<InputType>()->get() + numElementsPerEvent*(m_lastEventIdxToKeep + 1);


		std::shared_ptr<Container<InputType> > ret = make_shared<Container<InputType> >(
			ContainerLocation::LocationGpu,
			inputData->getData<InputType>()->getStream(),
			ptrStart, ptrEnd, false);
		return ret;
	}

	shared_ptr<RecordObject> RxEventLimiterNode::checkTypeAndProcess(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<USRawData> pRawData = nullptr;
		if (inObj && inObj->getType() == TypeUSRawData)
		{
			shared_ptr<USRawData> pInRawData = dynamic_pointer_cast<USRawData>(inObj);
			if (pInRawData)
			{
				// lock the object mutex to make sure no parameters are changed during processing
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();

				auto currentImageProperties = pInRawData->getImageProperties();
				auto currentRxBeamformerParameters = pInRawData->getRxBeamformerParameters();
				if (!m_lastSeenImageProperties ||
					!m_lastSeenRxBeamformerParameters ||
					m_lastSeenImageProperties != currentImageProperties ||
					m_lastSeenRxBeamformerParameters != currentRxBeamformerParameters ||
					m_parametersRequireUpdate)
				{
					updateImagePropertiesAndRxBeamformerParameters(
						currentImageProperties,
						currentRxBeamformerParameters);
					m_parametersRequireUpdate = false;
				}

				std::shared_ptr<ContainerBase> pRawDataProcessedData;

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types, 
				// only those meaningful for the operation of the node.
				switch (pInRawData->getDataType())
				{
				case TypeUint8:
					pRawDataProcessedData = processTemplateSelection<uint8_t>(pInRawData);
					break;
				case TypeInt16:
					pRawDataProcessedData = processTemplateSelection<int16_t>(pInRawData);
					break;
				case TypeFloat:
					pRawDataProcessedData = processTemplateSelection<float>(pInRawData);
					break;
				default:
					logging::log_error("RxEventLimiterNode: Input rawData type not supported");
					break;
				}
				m_callFrequency.measureEnd();

				// Wrap the returned Container in an USRawData with the same parameters etc. (except scanlines)
				pRawData = std::make_shared<USRawData>(
					m_modifiedImageProperties->getNumScanlines(),
					pInRawData->getNumElements(),
					pInRawData->getElementLayout(),
					pInRawData->getNumReceivedChannels(),
					pInRawData->getNumSamples(),
					pInRawData->getSamplingFrequency(),
					pRawDataProcessedData,
					m_modifiedRxBeamformerParameters,
					m_modifiedImageProperties,
					pInRawData->getReceiveTimestamp(),
					pInRawData->getSyncTimestamp());
			}
			else {
				logging::log_error("RxEventLimiterNode: could not cast object to USRawData type, is it in suppored ElementType?");
			}
		}
		return pRawData;
	}

	void RxEventLimiterNode::updateImagePropertiesAndRxBeamformerParameters(
		std::shared_ptr<const USImageProperties> newImageProperties,
		std::shared_ptr<const RxBeamformerParameters> newRxBeamformerParameters)
	{
		m_modifiedImageProperties = std::make_shared<USImageProperties>(*newImageProperties);
		m_modifiedRxBeamformerParameters = std::make_shared<RxBeamformerParameters>(*newRxBeamformerParameters);

		auto scanlineInfo = m_modifiedImageProperties->getScanlineInfo();
		auto scanlineLayout = m_modifiedImageProperties->getScanlineLayout();

		// Verify that the node parameters make sense
		m_lastEventIdxToKeep = std::min(m_lastEventIdxToKeep, static_cast<uint32_t>(scanlineLayout.x - 1));
		m_firstEventIdxToKeep = std::min(m_firstEventIdxToKeep, static_cast<uint32_t>(scanlineLayout.x - 1));

		// create new scanline layout
		scanlineLayout.x = m_lastEventIdxToKeep - m_firstEventIdxToKeep + 1;
		m_modifiedImageProperties->setScanlineLayout(scanlineLayout);

		// remove rx scanline infos accordingly
		// count the rx scanline infos that use only the remaining events
		size_t numRxScanlines = 0;
		for (size_t rxScanlineIdxX = 0; rxScanlineIdxX < scanlineInfo->size(); rxScanlineIdxX++)
		{
			bool canKeepScanline = true;
			for (size_t rxScanlineIdxY = 0; rxScanlineIdxY < (*scanlineInfo)[0].size(); rxScanlineIdxY++)
			{
				for (size_t txScanlineIdx = 0; txScanlineIdx < std::extent<decltype((*scanlineInfo)[rxScanlineIdxX][0].txParameters)>::value; txScanlineIdx++)
				{
					canKeepScanline = canKeepScanline &&
						(*scanlineInfo)[rxScanlineIdxX][rxScanlineIdxY].txParameters[txScanlineIdx].txScanlineIdx >= m_firstEventIdxToKeep &&
						(*scanlineInfo)[rxScanlineIdxX][rxScanlineIdxY].txParameters[txScanlineIdx].txScanlineIdx <= m_lastEventIdxToKeep;
				}
			}
			if (canKeepScanline)
			{
				numRxScanlines++;
			}
		}
		if (numRxScanlines == 0)
		{
			logging::log_error("RxEventLimiterNode: Selected rx events require removal of all rxScanlines.");
			m_lastEventIdxToKeep = static_cast<uint32_t>(scanlineLayout.x - 1);
			m_firstEventIdxToKeep = 0;
			updateImagePropertiesAndRxBeamformerParameters(newImageProperties, newRxBeamformerParameters);
		}
		else {

			// copy the respective rxScanlines
			decltype(scanlineInfo) newScanlineInfo = std::make_shared<std::vector<std::vector<ScanlineRxParameters3D> > >(numRxScanlines);
			size_t numRxScanlinesCopied = 0;
			for (size_t rxScanlineIdxX = 0; rxScanlineIdxX < scanlineInfo->size(); rxScanlineIdxX++)
			{
				bool canKeepScanline = true;
				for (size_t rxScanlineIdxY = 0; rxScanlineIdxY < (*scanlineInfo)[0].size(); rxScanlineIdxY++)
				{
					for (size_t txScanlineIdx = 0; txScanlineIdx < std::extent<decltype((*scanlineInfo)[rxScanlineIdxX][0].txParameters)>::value; txScanlineIdx++)
					{
						canKeepScanline = canKeepScanline &&
							(*scanlineInfo)[rxScanlineIdxX][rxScanlineIdxY].txParameters[txScanlineIdx].txScanlineIdx >= m_firstEventIdxToKeep &&
							(*scanlineInfo)[rxScanlineIdxX][rxScanlineIdxY].txParameters[txScanlineIdx].txScanlineIdx <= m_lastEventIdxToKeep;
					}
				}
				if (canKeepScanline)
				{
					auto rxScanlinesOriginal = (*scanlineInfo)[rxScanlineIdxX];
					(*newScanlineInfo)[numRxScanlinesCopied].resize(rxScanlinesOriginal.size());
					for (size_t rxScanlineIdxY = 0; rxScanlineIdxY < rxScanlinesOriginal.size(); rxScanlineIdxY++)
					{
						(*newScanlineInfo)[numRxScanlinesCopied][rxScanlineIdxY] = rxScanlinesOriginal[rxScanlineIdxY];

						//Fix the txIndex!
						for (size_t txScanlineIdx = 0;
							txScanlineIdx < std::extent<decltype((*newScanlineInfo)[numRxScanlinesCopied][rxScanlineIdxY].txParameters)>::value;
							txScanlineIdx++)
						{
							(*newScanlineInfo)[numRxScanlinesCopied][rxScanlineIdxY].txParameters[txScanlineIdx].txScanlineIdx -= m_firstEventIdxToKeep;
						}
					}
					numRxScanlinesCopied++;
				}
			}
			m_modifiedImageProperties->setScanlineInfo(newScanlineInfo);

			std::vector<ScanlineRxParameters3D> newScanlineInfoLinear(newScanlineInfo->size()* (*newScanlineInfo)[0].size());
			size_t scanlineIdxLinear = 0;
			for (size_t scanlineIdxX = 0; scanlineIdxX < newScanlineInfo->size(); scanlineIdxX++)
			{
				for (size_t scanlineIdxY = 0; scanlineIdxY < (*newScanlineInfo)[0].size(); scanlineIdxY++)
				{
					newScanlineInfoLinear[scanlineIdxLinear] = (*newScanlineInfo.get())[scanlineIdxX][scanlineIdxY];
					scanlineIdxLinear++;
				}
			}

			m_modifiedRxBeamformerParameters = std::make_shared<RxBeamformerParameters>(
				newScanlineInfo->size()* (*newScanlineInfo)[0].size(),
				vec2s{ newScanlineInfo->size(), (*newScanlineInfo)[0].size() },
				newRxBeamformerParameters->getSpeedOfSoundMMperS(),
				newRxBeamformerParameters->getRxDepths(),
				newScanlineInfoLinear,
				newRxBeamformerParameters->getRxElementXs(),
				newRxBeamformerParameters->getRxElementYs(),
				newRxBeamformerParameters->getRxNumDepths());

			m_lastSeenImageProperties = newImageProperties;
			m_lastSeenRxBeamformerParameters = newRxBeamformerParameters;
		}
	}
}