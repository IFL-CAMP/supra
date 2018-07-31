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
	BeamformingNode::BeamformingNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_lastSeenImageProperties(nullptr)
		, m_beamformer(nullptr)
		, m_lastSeenBeamformerParameters(nullptr)
	{
		

		if (queueing)
		{
			m_node = unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); }));
		}
		else
		{
			m_node = unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); }));
		}

		m_callFrequency.setName("Beamforming");
		m_valueRangeDictionary.set<double>("fNumber", 0.1, 4, 1, "F-Number");
		m_valueRangeDictionary.set<string>("windowType", { "Rectangular", "Hann", "Hamming", "Gauss" }, "Rectangular", "RxWindow");
		m_valueRangeDictionary.set<double>("windowParameter", 0.0, 10.0, 0.0, "RxWindow parameter");
		m_valueRangeDictionary.set<string>("beamformerType", { "DelayAndSum", "DelayAndStdDev", "TestSignal"}, "DelayAndSum", "RxBeamformer");
		m_valueRangeDictionary.set<bool>("interpolateTransmits", { false, true }, false, "Interpolate Transmits");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeInt16 }, TypeFloat, "Output type");
		configurationChanged();
		
	}

	void BeamformingNode::configurationChanged()
	{
		m_fNumber = m_configurationDictionary.get<double>("fNumber");
		readWindowType();
		m_windowParameter = m_configurationDictionary.get<double>("windowParameter");
		readBeamformerType();
		m_interpolateTransmits = m_configurationDictionary.get<bool>("interpolateTransmits");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
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
		else if (configKey == "beamformerType")
		{
			readBeamformerType();
		}
		else if (configKey == "interpolateTransmits")
		{
			m_interpolateTransmits = m_configurationDictionary.get<bool>("interpolateTransmits");
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

	template <typename InputType>
	std::shared_ptr<USImage> BeamformingNode::beamformTemplated(std::shared_ptr<const USRawData> pRawData)
	{
		switch (m_outputType)
		{
		case supra::TypeInt16:
			return m_beamformer->performRxBeamforming<InputType, int16_t>(
				m_beamformerType, pRawData, m_fNumber,
				m_windowType, static_cast<WindowFunction::ElementType>(m_windowParameter), m_interpolateTransmits);
			break;
		case supra::TypeFloat:
			return m_beamformer->performRxBeamforming<InputType, float>(
				m_beamformerType, pRawData, m_fNumber,
				m_windowType, static_cast<WindowFunction::ElementType>(m_windowParameter), m_interpolateTransmits);
			break;
		default:
			logging::log_error("BeamformingNode: Output image type not supported");
			break;
		}
		return nullptr;
	}

	shared_ptr<RecordObject> BeamformingNode::checkTypeAndBeamform(shared_ptr<RecordObject> inObj)
	{
		unique_lock<mutex> l(m_mutex);

		typedef std::chrono::high_resolution_clock Clock;
		typedef std::chrono::milliseconds milliseconds;
		Clock::time_point t0 = Clock::now();

		shared_ptr<USImage> pImageRF = nullptr;
		if (inObj->getType() == TypeUSRawData)
		{
			shared_ptr<const USRawData> pRawData = dynamic_pointer_cast<const USRawData>(inObj);
			if (pRawData)
			{
				m_callFrequency.measure();
				
				// We need to create a new beamformer if we either did not create one yet, or if the beamformer parameters changed
				bool needNewBeamformer = !m_beamformer;
				if (!m_lastSeenBeamformerParameters || m_lastSeenBeamformerParameters != pRawData->getRxBeamformerParameters())
				{
					m_lastSeenBeamformerParameters = pRawData->getRxBeamformerParameters();
					needNewBeamformer = true;
				}
				if (needNewBeamformer)
				{
					m_beamformer = std::make_shared<RxBeamformerCuda>(*m_lastSeenBeamformerParameters);
				}

				switch (pRawData->getDataType())
				{
				case TypeFloat:
					pImageRF = beamformTemplated<float>(pRawData);
					break;
				case TypeInt16:
					pImageRF = beamformTemplated<int16_t>(pRawData);
					break;
				default:
					logging::log_error("BeamformingNode: Input rawdata type not supported");
				}

				m_callFrequency.measureEnd();

				if (m_lastSeenImageProperties != pImageRF->getImageProperties())
				{
					updateImageProperties(pImageRF->getImageProperties());
				}
				pImageRF->setImageProperties(m_editedImageProperties);
				//logging::log_log("BF: ", pImageRF->getSize().x, " ", pImageRF->getSize().y, " ", pImageRF->getSize().z);
			}
			else {
				logging::log_error("BeamformingNode: could not cast object to USRawData type, is it in supported ElementType?");
			}
		}

		Clock::time_point t1 = Clock::now();
		milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
		std::cout << "Time to Beam Form: " << ms.count() << "ms\n";

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

	void BeamformingNode::readBeamformerType()
	{
		string beamformer = m_configurationDictionary.get<string>("beamformerType");
		m_beamformerType = RxBeamformerCuda::DelayAndSum;
		if (beamformer == "DelayAndSum")
		{
			m_beamformerType = RxBeamformerCuda::DelayAndSum;
		}
		else if (beamformer == "DelayAndStdDev")
		{
			m_beamformerType = RxBeamformerCuda::DelayAndStdDev;
		}
		else if (beamformer == "TestSignal")
		{
			m_beamformerType = RxBeamformerCuda::TestSignal;
		}
	}

	void BeamformingNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		m_editedImageProperties = make_shared<USImageProperties>(*imageProperties);
		m_editedImageProperties->setSpecificParameter("Beamformer.FNumber", m_fNumber);
		m_editedImageProperties->setSpecificParameter("Beamformer.WindowType", m_windowType);
		m_editedImageProperties->setSpecificParameter("Beamformer.WindowParameter", m_windowParameter);
		m_editedImageProperties->setSpecificParameter("Beamformer.BeamformerType", m_beamformerType);
		m_editedImageProperties->setSpecificParameter("Beamformer.InterpolateTransmits", m_interpolateTransmits);
	}
}