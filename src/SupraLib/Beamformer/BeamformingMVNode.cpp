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
	BeamformingMVNode::BeamformingMVNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_node(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); })
		, m_lastSeenImageProperties(nullptr)
	{
		m_callFrequency.setName("BeamformingMV");
		m_valueRangeDictionary.set<uint32_t>("subArraySize", 0, 64, 0, "Sub-array size");
		m_valueRangeDictionary.set<uint32_t>("temporalSmoothing", 0, 10, 3, "temporal smoothing");
		
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
		if (m_lastSeenImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}

	shared_ptr<RecordObject> BeamformingMVNode::checkTypeAndBeamform(shared_ptr<RecordObject> inObj)
	{
		unique_lock<mutex> l(m_mutex);

		shared_ptr<USImage<int16_t> > pImageRF = nullptr;
		if (inObj->getType() == TypeUSRawData)
		{
			shared_ptr<const USRawData<int16_t> > pRawData = dynamic_pointer_cast<const USRawData<int16_t>>(inObj);
			if (pRawData)
			{
				if (pRawData->getImageProperties()->getImageState() == USImageProperties::RawDelayed)
				{
					m_callFrequency.measure();

					cudaSafeCall(cudaDeviceSynchronize());

					cublasSafeCall(cublasSetStream(m_cublasH, pRawData->getData()->getStream()));
					pImageRF = performRxBeamforming<int16_t, int16_t>(
						pRawData, m_subArraySize, m_temporalSmoothing, m_cublasH);
					m_callFrequency.measureEnd();
					cudaSafeCall(cudaDeviceSynchronize());

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
