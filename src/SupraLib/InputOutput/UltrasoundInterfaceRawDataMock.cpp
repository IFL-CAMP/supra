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
#include "Beamformer/USRawData.h"
#include "UltrasoundInterfaceRawDataMock.h"
#include "utilities/utility.h"

#include "Beamformer/RxBeamformerParameters.h"
#include "ContainerFactory.h"

#include <memory>

using namespace std;

namespace supra
{
	UltrasoundInterfaceRawDataMock::UltrasoundInterfaceRawDataMock(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractInput<RecordObject>(graph, nodeID)
		, m_sequenceIndex(0)
		, m_frameIndex(0)
		, m_numel(0)
		, m_frozen(false)
	{
		m_callFrequency.setName("RawMock");
		//Setup allowed values for parameters
		m_valueRangeDictionary.set<bool>("singleImage", { true, false }, false, "Single image");
		m_valueRangeDictionary.set<bool>("streamSequenceOnce", { true, false }, false, "Emit sequences once");
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

		m_protoRawData = RxBeamformerParameters::readMetaDataForMock<int16_t>(m_mockMetadataFilename);

		m_numel = m_protoRawData->getNumReceivedChannels()*m_protoRawData->getNumSamples()*m_protoRawData->getNumScanlines();

		// initialize m_mockDataStreams and m_sequenceLengths by getting the file sizes of all datafiles
		m_mockDataStramReadBuffers.resize(m_mockDataFilenames.size());
		m_mockDataStreams.resize(m_mockDataFilenames.size());
		m_sequenceLengths.resize(m_mockDataFilenames.size());
		for (size_t k = 0; k < m_mockDataFilenames.size(); k++)
		{
			// In order to maximize reading performance, the ifstream needs a large read buffer
			m_mockDataStramReadBuffers[k].resize(128 * 1024 * 1024, '\0');
			m_mockDataStreams[k] = std::shared_ptr<std::ifstream>(new std::ifstream);
			m_mockDataStreams[k]->open(m_mockDataFilenames[k], std::ifstream::ate | std::ifstream::binary);
			m_mockDataStreams[k]->rdbuf()->pubsetbuf(m_mockDataStramReadBuffers[k].data(), m_mockDataStramReadBuffers[k].size());	
			size_t filesizeBytes = m_mockDataStreams[k]->tellg();
			m_mockDataStreams[k]->seekg(0);

			m_sequenceLengths[k] = filesizeBytes / (m_numel * sizeof(int16_t));
		}

		readNextFrame();
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
		if (configKey == "sequenceOnce")
		{
			m_streamSequenceOnce = m_configurationDictionary.get<bool>("streamSequenceOnce");
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

			m_callFrequency.measure();
			shared_ptr<USRawData<int16_t> > pRawData = std::make_shared<USRawData<int16_t> >(
				m_protoRawData->getNumScanlines(),
				m_protoRawData->getNumElements(),
				m_protoRawData->getElementLayout(),
				m_protoRawData->getNumReceivedChannels(),
				m_protoRawData->getNumSamples(),
				m_protoRawData->getSamplingFrequency(),
				m_pMockData,
				m_protoRawData->getRxBeamformerParameters(),
				m_protoRawData->getImageProperties(),
				getCurrentTime(),
				getCurrentTime());
			addData<0>(pRawData);

			if (!m_singleImage)
			{
				readNextFrame();
			}
			m_callFrequency.measureEnd();
		}
		return getRunning();
	}

	void UltrasoundInterfaceRawDataMock::readConfiguration()
	{
		lock_guard<mutex> lock(m_objectMutex);
		//read conf values
		m_singleImage = m_configurationDictionary.get<bool>("singleImage");
		m_streamSequenceOnce = m_configurationDictionary.get<bool>("streamSequenceOnce");
		m_frequency = m_configurationDictionary.get<int>("frequency");
		m_mockMetadataFilename = m_configurationDictionary.get<string>("mockMetaDataFilename");
		m_mockDataFilenames = split(m_configurationDictionary.get<string>("mockDataFilename"), ',');
		for (auto& filename : m_mockDataFilenames)
		{
			filename = trim(filename);
		}
	}

	void UltrasoundInterfaceRawDataMock::readNextFrame()
	{
		auto mockDataHost = make_shared<Container<int16_t> >(LocationHost, ContainerFactory::getNextStream(), m_numel);

		m_mockDataStreams[m_sequenceIndex]->read(reinterpret_cast<char*>(mockDataHost->get()), m_numel * sizeof(int16_t));
		m_pMockData = make_shared<Container<int16_t> >(LocationGpu, *mockDataHost);
		// advance to the next image and sequence where required
		m_frameIndex = (m_frameIndex + 1) % m_sequenceLengths[m_sequenceIndex];
		if (m_frameIndex == 0)
		{
			m_mockDataStreams[m_sequenceIndex]->seekg(0);
			m_sequenceIndex = (m_sequenceIndex + 1) % m_sequenceLengths.size();
			if (m_sequenceIndex == 0 && m_streamSequenceOnce)
			{
				setRunning(false);
			}
		}
	}

	bool UltrasoundInterfaceRawDataMock::ready()
	{
		return true;
	}
}