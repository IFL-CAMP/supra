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

#include <memory>

#include "USImage.h"
#include "Beamformer/USRawData.h"
#include "UltrasoundInterfaceRawDataMock.h"
#include "utilities/utility.h"

#include "Beamformer/RxBeamformerCuda.h"

using namespace std;

namespace supra
{
	UltrasoundInterfaceRawDataMock::UltrasoundInterfaceRawDataMock(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractInput<RecordObject>(graph, nodeID)
		, m_sequenceIndex(0)
		, m_sequenceLength(0)
		, m_numel(0)
		, m_frozen(false)
	{
		m_callFrequency.setName("RawMock");
		//Setup allowed values for parameters
		m_valueRangeDictionary.set<bool>("singleImage", { true, false }, false, "Single image");
		m_valueRangeDictionary.set<int>("frequency", 1, 100, 5, "Frequency");
		m_valueRangeDictionary.set<string>("mockMetaDataFilename", "", "Mock meta data filename");
		m_valueRangeDictionary.set<string>("mockDataFilename", "", "Mock data filename");

		readConfiguration();
	}

	void UltrasoundInterfaceRawDataMock::initializeDevice()
	{
		if (getTimerFrequency() != m_frequency)
		{
			setUpTimer(m_frequency);
		}

		m_protoRawData = RxBeamformerCuda::readMetaDataForMock<int16_t>(m_mockMetadataFilename);

		m_numel = m_protoRawData->getNumReceivedChannels()*m_protoRawData->getNumSamples()*m_protoRawData->getNumScanlines();

		m_pMockData = make_shared<Container<int16_t> >(LocationHost, min(m_maxSequenceLength * m_numel, m_maxSequenceSizeMb * 1024 * 1024 / sizeof(int16_t)));

		ifstream f(m_mockDataFilename, std::ios::binary);
		f.read(reinterpret_cast<char*>(m_pMockData->get()), min(m_maxSequenceLength * m_numel * sizeof(int16_t), m_maxSequenceSizeMb * 1024 * 1024));
		size_t numBytesRead = f.gcount();

		m_sequenceLength = numBytesRead / (m_numel * sizeof(int16_t));

		m_pMockData = make_shared<Container<int16_t> >(LocationHost, m_pMockData->get(), m_pMockData->get() + m_sequenceLength*m_numel);
	}

	void UltrasoundInterfaceRawDataMock::freeze()
	{
		m_frozen = true;
	}

	void UltrasoundInterfaceRawDataMock::unfreeze()
	{
		m_frozen = false;
	}

	void UltrasoundInterfaceRawDataMock::startAcquisition()
	{
		setUpTimer(m_frequency);
		timerLoop();
	}

	void UltrasoundInterfaceRawDataMock::configurationEntryChanged(const std::string & configKey)
	{
		lock_guard<mutex> lock(m_objectMutex);
		if (configKey == "frequency")
		{
			m_frequency = m_configurationDictionary.get<int>("frequency");
			if (getTimerFrequency() != m_frequency)
			{
				setUpTimer(m_frequency);
			}
		}
		if (configKey == "singleImage")
		{
			m_singleImage = m_configurationDictionary.get<bool>("singleImage");
		}
	}

	void UltrasoundInterfaceRawDataMock::configurationChanged()
	{
		readConfiguration();
	}

	bool UltrasoundInterfaceRawDataMock::timerCallback() {
		if (!m_frozen)
		{
			double timestamp = getCurrentTime();

			size_t imageIndex = m_sequenceIndex;
			if (m_singleImage)
			{
				imageIndex = 0;
			}
			auto dataGpu = make_shared<Container<int16_t> >(LocationGpu,
				m_pMockData->get() + imageIndex     *m_numel,
				m_pMockData->get() + (imageIndex + 1)*m_numel);

			m_callFrequency.measure();
			shared_ptr<USRawData<int16_t> > pRawData = std::make_shared<USRawData<int16_t> >(
				m_protoRawData->getNumScanlines(),
				m_protoRawData->getNumElements(),
				m_protoRawData->getElementLayout(),
				m_protoRawData->getNumReceivedChannels(),
				m_protoRawData->getNumSamples(),
				m_protoRawData->getSamplingFrequency(),
				dataGpu,
				m_protoRawData->getRxBeamformer(),
				m_protoRawData->getImageProperties(),
				getCurrentTime(),
				getCurrentTime());
			addData<0>(pRawData);

			m_sequenceIndex = (m_sequenceIndex + 1) % m_sequenceLength;
		}
		return getRunning();
	}

	void UltrasoundInterfaceRawDataMock::readConfiguration()
	{
		lock_guard<mutex> lock(m_objectMutex);
		//read conf values
		m_singleImage = m_configurationDictionary.get<bool>("singleImage");
		m_frequency = m_configurationDictionary.get<int>("frequency");
		m_mockMetadataFilename = m_configurationDictionary.get<string>("mockMetaDataFilename");
		m_mockDataFilename = m_configurationDictionary.get<string>("mockDataFilename");
	}

	bool UltrasoundInterfaceRawDataMock::ready()
	{
		return true;
	}
}