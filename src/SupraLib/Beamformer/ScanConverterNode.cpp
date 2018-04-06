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

#include "ScanConverterNode.h"
#include "ScanConverter.h"

#include "USImage.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	ScanConverterNode::ScanConverterNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_node(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndConvert(inObj); })
		, m_maskOutputNode(graph)
		, m_maskSent(false)
		, m_parameterChangeRequiresInternalUpdate(false)
	{
		m_converter = unique_ptr<ScanConverter>(new ScanConverter());
		m_callFrequency.setName("ScanConv");

		m_valueRangeDictionary.set<double>("imageResolution", 0.01, 5, 0.1, "Forced image resolution");
		m_valueRangeDictionary.set<bool>("imageResolutionForced", { true, false }, false, "Force image resolution");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeInt16, TypeUint8}, TypeFloat, "Output type");

		configurationChanged();
	}

	void ScanConverterNode::configurationChanged()
	{
		m_imageResolution = m_configurationDictionary.get<double>("imageResolution");
		m_forceImageResolution = m_configurationDictionary.get<bool>("imageResolutionForced");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
	}

	void ScanConverterNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "outputType")
		{
			m_outputType = m_configurationDictionary.get<DataType>("outputType");
		}
		else
		{
			m_imageResolution = m_configurationDictionary.get<double>("imageResolution");
			m_forceImageResolution = m_configurationDictionary.get<bool>("imageResolutionForced");

			m_parameterChangeRequiresInternalUpdate = true;
			logging::log_log_if(m_parameterChangeRequiresInternalUpdate,
				"ScanConverterNode: Update of internals required because scanconversion parameters have changed.");
		}
	}

	void ScanConverterNode::sendMask(shared_ptr<RecordObject> pImage)
	{
		auto mask = m_converter->getMask();
		auto maskImage =
			make_shared<USImage>(
				m_converter->getImageSize(),
				mask,
				m_scanConvImageProperties,
				pImage->getReceiveTimestamp(),
				pImage->getSyncTimestamp());

		m_maskOutputNode.try_put(maskImage);
		m_maskSent = true;
	}

	template <typename T>
	shared_ptr<RecordObject> ScanConverterNode::convertTemplated(shared_ptr<USImage> pInImage)
	{
		m_callFrequency.measure();
		std::shared_ptr<ContainerBase> pImageData;
		switch (m_outputType)
		{
		case TypeFloat:
			pImageData = m_converter->convert<T, float>(pInImage);
			break;
		case TypeInt16:
			pImageData = m_converter->convert<T, int16_t>(pInImage);
			break;
		case TypeUint8:
			pImageData = m_converter->convert<T, uint8_t>(pInImage);
			break;
		default:
			logging::log_error("ScanConverterNode: Image output type not supported");
			break;
		}
		m_callFrequency.measureEnd();

		return make_shared<USImage>(
			m_converter->getImageSize(),
			pImageData,
			m_scanConvImageProperties,
			pInImage->getReceiveTimestamp(),
			pInImage->getSyncTimestamp());
	}

	shared_ptr<RecordObject> ScanConverterNode::checkTypeAndConvert(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<RecordObject> pImage = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
			if (pInImage)
			{
				unique_lock<mutex> l(m_mutex);
				shared_ptr<const USImageProperties> currentProperties = pInImage->getImageProperties();

				bool internalUpdateNeeded = m_parameterChangeRequiresInternalUpdate;
				if (currentProperties != m_lastSeenImageProperties)
				{
					bool imagePropertiesChanged = !m_lastSeenImageProperties ||
						!(m_lastSeenImageProperties->getSampleDistance() == currentProperties->getSampleDistance() &&
							m_lastSeenImageProperties->getNumScanlines() == currentProperties->getNumScanlines() &&
							m_lastSeenImageProperties->is2D() == currentProperties->is2D() &&
							m_lastSeenImageProperties->getScanlineLayout() == currentProperties->getScanlineLayout() &&
							m_lastSeenImageProperties->getDepth() == currentProperties->getDepth() &&
							m_lastSeenImageProperties->getImageResolution() == currentProperties->getImageResolution() &&
							m_lastSeenImageProperties->getScanlineInfo() == currentProperties->getScanlineInfo());
					if (imagePropertiesChanged)
					{
						logging::log_log_if(!m_lastSeenImageProperties, "ScanConverterNode: Update of internals required because we received the first image properties.");
						if (m_lastSeenImageProperties)
						{
							logging::log_log_if(imagePropertiesChanged, "ScanConverterNode: Update of internals required because the image properties have changed. ",
								"(difference in sampleDistance: ", m_lastSeenImageProperties->getSampleDistance() != currentProperties->getSampleDistance(),
								", numScanlines: ", m_lastSeenImageProperties->getNumScanlines() != currentProperties->getNumScanlines(),
								", is2D: ", m_lastSeenImageProperties->is2D() != currentProperties->is2D(),
								", scanlineLayout: ", !(m_lastSeenImageProperties->getScanlineLayout() == currentProperties->getScanlineLayout()),
								", depth: ", m_lastSeenImageProperties->getDepth() != currentProperties->getDepth(),
								", imageResolution: ", m_lastSeenImageProperties->getImageResolution() != currentProperties->getImageResolution(),
								", rxScanlines: ", !(m_lastSeenImageProperties->getScanlineInfo() == currentProperties->getScanlineInfo()), ")");
						}
					}

					internalUpdateNeeded = internalUpdateNeeded || imagePropertiesChanged;

					m_lastSeenImageProperties = currentProperties;
					m_scanConvImageProperties = make_shared<USImageProperties>(*m_lastSeenImageProperties);
					m_scanConvImageProperties->setImageState(USImageProperties::ImageState::Scan);

					if (m_forceImageResolution)
					{
						m_scanConvImageProperties->setImageResolution(m_imageResolution);
					}
				}

				if (internalUpdateNeeded)
				{
					m_parameterChangeRequiresInternalUpdate = false;
					if (m_forceImageResolution)
					{
						m_scanConvImageProperties->setImageResolution(m_imageResolution);
					}
					else if (m_scanConvImageProperties->getImageResolution() != m_lastSeenImageProperties->getImageResolution())
					{
						m_scanConvImageProperties->setImageResolution(m_lastSeenImageProperties->getImageResolution());
					}

					m_converter->updateInternals(m_scanConvImageProperties);
					m_maskSent = false;
				}

				switch (pInImage->getDataType())
				{
				case TypeFloat:
					pImage = convertTemplated<float>(pInImage);
					break;
				case TypeInt16:
					pImage = convertTemplated<int16_t>(pInImage);
					break;
				case TypeUint8:
					pImage = convertTemplated<uint8_t>(pInImage);
					break;
				default:
					logging::log_error("ScanConverterNode: Image input type not supported");
					break;
				}

				if (!m_maskSent)
				{
					sendMask(inObj);
				}
			}
			else {
				logging::log_error("ScanConverterNode: could not cast object to USImage type");
			}
		}
		return pImage;
	}
}