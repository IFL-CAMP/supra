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


#include "TrackerInterfaceSimulated.h"
#include <utilities/utility.h>

using namespace std;

namespace supra
{
	void TrackerInterfaceSimulated::freeze()
	{
		m_frozen = true;
	}
	void TrackerInterfaceSimulated::unfreeze()
	{
		m_frozen = false;
	}

	void TrackerInterfaceSimulated::startAcquisition()
	{
		m_callFrequency.setName("TR");
		setUpTimer(m_frequency);
		timerLoop();
	}

	void TrackerInterfaceSimulated::configurationEntryChanged(const std::string & configKey)
	{
		readConfiguration();
	}

	void TrackerInterfaceSimulated::configurationChanged()
	{
		readConfiguration();
	}

	bool TrackerInterfaceSimulated::timerCallback()
	{
		if (!m_frozen)
		{
			double timestamp = getCurrentTime();
			shared_ptr<TrackerDataSet> pTrackingDataSet;
			{
				lock_guard<mutex> lock(m_objectMutex);
				m_callFrequency.measure();

				std::vector<TrackerData> trackingData;
				double currentTime = getCurrentTime();

				trackingData.push_back(TrackerData(0, 0, m_currentZ, 0, 0, 0, 0, 100, 666, "SimulTranslation", currentTime));
				m_currentZ += 0.02;

				pTrackingDataSet = make_shared<TrackerDataSet>(trackingData, currentTime, currentTime);
			}
			addData(0, pTrackingDataSet);
		}
		return getRunning();
	}

	void TrackerInterfaceSimulated::readConfiguration()
	{
		std::lock_guard<std::mutex> lock(m_objectMutex);
		//read conf values
		m_frequency = m_configurationDictionary.get<int>("frequency", 30);

		if (getTimerFrequency() != m_frequency)
		{
			setUpTimer(m_frequency);
		}
	}
}