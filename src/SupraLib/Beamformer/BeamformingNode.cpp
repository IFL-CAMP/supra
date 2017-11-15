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

#include "BeamformingNode.h"

#include "USImage.h"
#include "USRawData.h"
#include "RxBeamformerCuda.h"

#include <utilities/Logging.h>
#include <algorithm>
using namespace std;

namespace supra
{
	BeamformingNode::BeamformingNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_node(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); })
		, m_lastSeenImageProperties(nullptr)
	{
		m_callFrequency.setName("Beamforming");
		m_valueRangeDictionary.set<double>("fNumber", 0.1, 4, 1, "F-Number");
		m_valueRangeDictionary.set<string>("windowType", { "Rectangular", "Hann", "Hamming", "Gauss" }, "Rectangular", "RxWindow");
		m_valueRangeDictionary.set<double>("windowParameter", 0.0, 10.0, 0.0, "RxWindow parameter");
		m_valueRangeDictionary.set<bool>("interpolateTransmits", { false, true }, false, "Interpolate Transmits");
		m_valueRangeDictionary.set<bool>("testSignal", { false, true }, false, "Test signal");
		configurationChanged();
	}

	void BeamformingNode::configurationChanged()
	{
		m_fNumber = m_configurationDictionary.get<double>("fNumber");
		readWindowType();
		m_windowParameter = m_configurationDictionary.get<double>("windowParameter");
		m_interpolateTransmits = m_configurationDictionary.get<bool>("interpolateTransmits");
		m_testSignal = m_configurationDictionary.get<bool>("testSignal");
	}

	void BeamformingNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "fNumber")
		{
			m_fNumber = m_configurationDictionary.get<double>("fNumber");
		}
		else if (configKey == "windowType")
		{
			readWindowType();
		}
		else if (configKey == "windowParameter")
		{
			m_windowParameter = m_configurationDictionary.get<double>("windowParameter");
		}
		else if (configKey == "interpolateTransmits")
		{
			m_interpolateTransmits = m_configurationDictionary.get<bool>("interpolateTransmits");
		}
		else if (configKey == "testSignal")
		{
			m_testSignal = m_configurationDictionary.get<bool>("testSignal");
		}
		if (m_lastSeenImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}

	shared_ptr<RecordObject> BeamformingNode::checkTypeAndBeamform(shared_ptr<RecordObject> inObj)
	{
		unique_lock<mutex> l(m_mutex);

		shared_ptr<USImage<int16_t> > pImageRF = nullptr;
		if (inObj->getType() == TypeUSRawData)
		{
			shared_ptr<const USRawData<int16_t> > pRawData = dynamic_pointer_cast<const USRawData<int16_t>>(inObj);
			if (pRawData)
			{
				m_callFrequency.measure();
				pImageRF = pRawData->getRxBeamformer()->performRxBeamforming<int16_t, int16_t>(
					pRawData, m_fNumber, m_windowType, static_cast<WindowFunction::ElementType>(m_windowParameter), m_interpolateTransmits, m_testSignal);
				m_callFrequency.measureEnd();

				if (m_lastSeenImageProperties != pImageRF->getImageProperties())
				{
					updateImageProperties(pImageRF->getImageProperties());
				}
				pImageRF->setImageProperties(m_editedImageProperties);
			}
			else {
				logging::log_error("BeamformingNode: could not cast object to USRawData type, is it in supported ElementType?");
			}
		}
		return pImageRF;
	}

	void BeamformingNode::readWindowType()
	{
		string window = m_configurationDictionary.get<string>("windowType");
		if (window == "Rectangular")
		{
			m_windowType = WindowRectangular;
		}
		else if (window == "Hann")
		{
			m_windowType = WindowHann;
		}
		else if (window == "Hamming")
		{
			m_windowType = WindowHamming;
		}
		else if (window == "Gauss")
		{
			m_windowType = WindowGauss;
		}
	}

	void BeamformingNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		m_editedImageProperties = make_shared<USImageProperties>(*imageProperties);
		m_editedImageProperties->setSpecificParameter("Beamformer.FNumber", m_fNumber);
		m_editedImageProperties->setSpecificParameter("Beamformer.WindowType", m_windowType);
		m_editedImageProperties->setSpecificParameter("Beamformer.WindowParameter", m_windowParameter);
		m_editedImageProperties->setSpecificParameter("Beamformer.InterpolateTransmits", m_interpolateTransmits);
		m_editedImageProperties->setSpecificParameter("Beamformer.TestSignal", m_testSignal);
	}
}