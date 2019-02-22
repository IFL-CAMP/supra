// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2017, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "BeamformingMVNode.h"
#include "RxBeamformerMV.h"

#include "USImage.h"
#include "USRawData.h"

#include <utilities/Logging.h>
#include <utilities/cudaUtility.h>
#include <utilities/cublasUtility.h>
using namespace std;

namespace supra
{
	BeamformingMVNode::BeamformingMVNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_lastSeenImageProperties(nullptr)
		, m_outputType(TypeUnknown)
	{
		if (queueing)
		{
			m_node = unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); }));
		}
		else
		{
			m_node = unique_ptr<NodeTypeDiscarding>(
				new NodeTypeDiscarding(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); }));
		}

		m_callFrequency.setName("BeamformingMV");
		m_valueRangeDictionary.set<uint32_t>("subArraySize", 0, 64, 0, "Sub-array size");
		m_valueRangeDictionary.set<uint32_t>("temporalSmoothing", 0, 10, 3, "temporal smoothing");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeUint16 }, TypeFloat, "Output type");
		m_valueRangeDictionary.set<double>("subArrayScalingPower", 0.5, 3.0, 1.5, "Subarray count scaling power");
		m_valueRangeDictionary.set<bool>("computeMeans", true, "compute signal means");
		
		configurationChanged();

		cublasSafeCall(cublasCreate(&m_cublasH));
		cublasSafeCall(cublasSetAtomicsMode(m_cublasH, CUBLAS_ATOMICS_ALLOWED));
	}

	BeamformingMVNode::~BeamformingMVNode()
	{
		cublasSafeCall(cublasDestroy(m_cublasH));
	}

	void BeamformingMVNode::configurationChanged()
	{
		m_subArraySize = m_configurationDictionary.get<uint32_t>("subArraySize");
		m_temporalSmoothing = m_configurationDictionary.get<uint32_t>("temporalSmoothing");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
		m_subArrayScalingPower = m_configurationDictionary.get<double>("subArrayScalingPower");
		m_computeMeans = m_configurationDictionary.get<bool>("computeMeans");
	}

	void BeamformingMVNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "subArraySize")
		{
			m_subArraySize = m_configurationDictionary.get<uint32_t>("subArraySize");
		}
		else if (configKey == "temporalSmoothing")
		{
			m_temporalSmoothing = m_configurationDictionary.get<uint32_t>("temporalSmoothing");
		}
		else if (configKey == "outputType")
		{
			m_outputType = m_configurationDictionary.get<DataType>("outputType");
		}
		else if (configKey == "subArrayScalingPower")
		{
			m_subArrayScalingPower = m_configurationDictionary.get<double>("subArrayScalingPower");
		}
		else if (configKey == "computeMeans")
		{
			m_computeMeans = m_configurationDictionary.get<bool>("computeMeans");
		}
		if (m_lastSeenImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}

	template <typename RawDataType>
	std::shared_ptr<USImage> BeamformingMVNode::beamformTemplated(shared_ptr<const USRawData> rawData)
	{
		shared_ptr<USImage> pImageRF = nullptr;
		cudaSafeCall(cudaDeviceSynchronize());
		cublasSafeCall(cublasSetStream(m_cublasH, rawData->getData<RawDataType>()->getStream()));
		switch (m_outputType)
		{
		case supra::TypeInt16:
			pImageRF = RxBeamformerMV::performRxBeamforming<RawDataType, int16_t>(
				rawData, m_subArraySize, m_temporalSmoothing, m_cublasH, m_subArrayScalingPower, m_computeMeans);
			break;
		case supra::TypeFloat:
			pImageRF = RxBeamformerMV::performRxBeamforming<RawDataType, float>(
				rawData, m_subArraySize, m_temporalSmoothing, m_cublasH, m_subArrayScalingPower, m_computeMeans);
			break;
		default:
			logging::log_error("BeamformingMVNode: Output image type not supported:");
			break;
		}
		cudaSafeCall(cudaDeviceSynchronize());
		return pImageRF;
	}

	shared_ptr<RecordObject> BeamformingMVNode::checkTypeAndBeamform(shared_ptr<RecordObject> inObj)
	{
		unique_lock<mutex> l(m_mutex);

		shared_ptr<USImage> pImageRF = nullptr;
		if (inObj->getType() == TypeUSRawData)
		{
			shared_ptr<const USRawData> pRawData = dynamic_pointer_cast<const USRawData>(inObj);
			if (pRawData)
			{
				if (pRawData->getImageProperties()->getImageState() == USImageProperties::RawDelayed)
				{
					m_callFrequency.measure();
					switch (pRawData->getDataType())
					{
					case TypeInt16:
						pImageRF = beamformTemplated<int16_t>(pRawData);
						break;
					case TypeFloat:
						pImageRF = beamformTemplated<float>(pRawData);
						break;
					default:
						logging::log_error("BeamformingMVNode: Input rawdata type is not supported.");
						break;
					}
					m_callFrequency.measureEnd();

					if (m_lastSeenImageProperties != pImageRF->getImageProperties())
					{
						updateImageProperties(pImageRF->getImageProperties());
					}
					pImageRF->setImageProperties(m_editedImageProperties);
				}
				else {
					logging::log_error("BeamformingMVNode: Cannot beamform undelayed RawData. Apply RawDelayNode first");
				}
			}
			else {
				logging::log_error("BeamformingMVNode: could not cast object to USRawData type, is it in supported ElementType?");
			}
		}
		return pImageRF;
	}

	void BeamformingMVNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		m_editedImageProperties = make_shared<USImageProperties>(*imageProperties);
		m_editedImageProperties->setImageState(USImageProperties::RF);
		m_editedImageProperties->setSpecificParameter("BeamformingMVNode.subArraySize", m_subArraySize);
		m_editedImageProperties->setSpecificParameter("BeamformingMVNode.temporalSmoothing", m_temporalSmoothing);
	}
}
