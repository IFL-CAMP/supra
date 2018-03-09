// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2011-2016, all rights reserved,
//      Christoph Hennersperger 
//		EmaiL christoph.hennersperger@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
//	and
//		Rüdiger Göbl
//		Email r.goebl@tum.de
//
// ================================================================================================

#include "USImage.h"
#include "UltrasoundInterfaceSimulated.h"
#include "utilities/utility.h"
#include "ContainerFactory.h"

#include <memory>

using namespace std;

namespace supra
{
	UltrasoundInterfaceSimulated::UltrasoundInterfaceSimulated(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractInput(graph, nodeID,1)
		, m_frozen(false)
	{
		m_callFrequency.setName("US-Sim");
		//Setup allowed values for parameters
		m_valueRangeDictionary.set<double>("gain", { 0.01, 0.1, 0.5, 1.0 }, 0.5, "Gain");
		m_valueRangeDictionary.set<size_t>("numVectors", 16, 256, 128, "Num of vectors (width)");
		m_valueRangeDictionary.set<size_t>("numSamples", 100, 1024, 600, "Num of samples (height)");
		m_valueRangeDictionary.set<int>("frequency", 5, 10000, 25, "Frequency");
		m_valueRangeDictionary.set<double>("depth", 20, 120, 60, "Depth (mm)");
		m_valueRangeDictionary.set<double>("width", 20, 60, 40, "Width (mm)");
	}

	void UltrasoundInterfaceSimulated::initializeDevice()
	{

	}

	void UltrasoundInterfaceSimulated::freeze()
	{
		m_frozen = true;
	}

	void UltrasoundInterfaceSimulated::unfreeze()
	{
		m_frozen = false;
	}

	void UltrasoundInterfaceSimulated::startAcquisition()
	{
		setUpTimer(m_frequency);
		timerLoop();
	}

	void UltrasoundInterfaceSimulated::configurationEntryChanged(const std::string & configKey)
	{
		readConfiguration();
	}

	void UltrasoundInterfaceSimulated::configurationChanged()
	{
		readConfiguration();
	}

	bool UltrasoundInterfaceSimulated::timerCallback() {
		if (!m_frozen)
		{
			double timestamp = getCurrentTime();
			//if (m_pCurrentContextMap) 
			shared_ptr<USImage<uint8_t> > pImage;
			{
				lock_guard<mutex> lock(m_objectMutex);
				m_callFrequency.measure();

				int modValue = rand() % std::min((int)(255 * m_gain), 255);

				auto pData = make_shared<Container<uint8_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), m_bmodeNumVectors*m_bmodeNumSamples);
				for (size_t k = 0; k < m_bmodeNumSamples; ++k)
				{
					uint8_t* dummyData = (pData->get()) + (k*m_bmodeNumVectors);
					for (unsigned int ss = 0; ss < m_bmodeNumVectors; ++ss)
						dummyData[ss] = modValue; //rand() % 
				}

				pImage = make_shared<USImage<uint8_t> >(
					vec2s{ m_bmodeNumVectors, m_bmodeNumSamples }, pData, m_pImageProperties, timestamp, timestamp);
				m_callFrequency.measureEnd();
			}
			addData<0>(pImage);
		}
		return getRunning();
	}

	void UltrasoundInterfaceSimulated::readConfiguration()
	{
		lock_guard<mutex> lock(m_objectMutex);
		//read conf values
		m_frequency = m_configurationDictionary.get<int>("frequency");
		m_bmodeNumVectors = m_configurationDictionary.get<size_t>("numVectors");
		m_bmodeNumSamples = m_configurationDictionary.get<size_t>("numSamples");
		m_gain = m_configurationDictionary.get<double>("gain");
		m_depth = m_configurationDictionary.get<double>("depth");
		m_width = m_configurationDictionary.get<double>("width");

		if (getTimerFrequency() != m_frequency)
		{
			setUpTimer(m_frequency);
		}

		//prepare the USImageProperties
		m_pImageProperties = make_shared<USImageProperties>(
			vec2s{ m_bmodeNumVectors, 1 },
			m_bmodeNumSamples,
			USImageProperties::ImageType::BMode,
			USImageProperties::ImageState::Scan,
			USImageProperties::TransducerType::Linear,
			m_depth);
	}

	bool UltrasoundInterfaceSimulated::ready()
	{
		return true;
	}
}
