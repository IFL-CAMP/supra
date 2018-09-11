#include "USImage.h"
#include "Beamformer/USRawData.h"
#include "UltrasoundInterfaceBeamformedMock.h"
#include "utilities/utility.h"

#include "Beamformer/RxBeamformerParameters.h"
#include "ContainerFactory.h"

#include <memory>

using namespace std;

namespace supra
{
	UltrasoundInterfaceBeamformedMock::UltrasoundInterfaceBeamformedMock(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractInput(graph, nodeID,1)
		, m_sequenceIndex(0)
		, m_frameIndex(0)
		, m_numel(0)
		, m_frozen(false)
		, m_lastFrame(false)
	{
		m_callFrequency.setName("BeamformedMock");
		//Setup allowed values for parameters
		m_valueRangeDictionary.set<bool>("singleImage", { true, false }, false, "Single image");
		m_valueRangeDictionary.set<bool>("streamSequenceOnce", { true, false }, false, "Emit sequences once");
		m_valueRangeDictionary.set<double>("frequency", 0.001, 100, 5, "Frequency");
		m_valueRangeDictionary.set<string>("mockMetaDataFilename", "", "Mock meta data filename");
		m_valueRangeDictionary.set<string>("mockDataFilename", "", "Mock data filename");

		readConfiguration();
	}

	void UltrasoundInterfaceBeamformedMock::initializeDevice()
	{
		if (getTimerFrequency() != m_frequency)
		{
			setUpTimer(m_frequency);
		}

		auto imageProps = make_shared<USImageProperties>(m_mockMetadataFilename);

		m_protoUSImage = make_shared<USImage>(
			vec2s{ imageProps->getNumScanlines(), imageProps->getNumSamples()},
			nullptr,
			imageProps,
			0, 0);

		auto size = m_protoUSImage->getSize();

		m_numel = size.x * size.y * size.z;

		// initialize m_mockDataStreams and m_sequenceLengths by getting the file sizes of all datafiles
		m_mockDataStramReadBuffers.resize(m_mockDataFilenames.size());
		m_mockDataStreams.resize(m_mockDataFilenames.size());
		m_sequenceLengths.resize(m_mockDataFilenames.size());
		for (size_t k = 0; k < m_mockDataFilenames.size(); k++)
		{
			// In order to maximize reading performance, the ifstream needs a large read buffer
			//m_mockDataStramReadBuffers[k].resize(128 * 1024 * 1024, '\0');
			m_mockDataStramReadBuffers[k].resize(128 * 1024, '\0');
			m_mockDataStreams[k] = std::shared_ptr<std::ifstream>(new std::ifstream);
			m_mockDataStreams[k]->open(m_mockDataFilenames[k], std::ifstream::ate | std::ifstream::binary);
			m_mockDataStreams[k]->rdbuf()->pubsetbuf(m_mockDataStramReadBuffers[k].data(), m_mockDataStramReadBuffers[k].size());	
			size_t filesizeBytes = m_mockDataStreams[k]->tellg();
			m_mockDataStreams[k]->seekg(0);

			m_sequenceLengths[k] = filesizeBytes / (m_numel * sizeof(int16_t));
		}

		readNextFrame();
	}

	void UltrasoundInterfaceBeamformedMock::freeze()
	{
		m_frozen = true;
	}

	void UltrasoundInterfaceBeamformedMock::unfreeze()
	{
		m_frozen = false;
	}

	void UltrasoundInterfaceBeamformedMock::startAcquisition()
	{
		setUpTimer(m_frequency);
		timerLoop();
	}

	void UltrasoundInterfaceBeamformedMock::configurationEntryChanged(const std::string & configKey)
	{
		lock_guard<mutex> lock(m_objectMutex);
		if (configKey == "frequency")
		{
			m_frequency = m_configurationDictionary.get<double>("frequency");
			if (getTimerFrequency() != m_frequency)
			{
				setUpTimer(m_frequency);
			}
		}
		if (configKey == "singleImage")
		{
			m_singleImage = m_configurationDictionary.get<bool>("singleImage");
		}
		if (configKey == "streamSequenceOnce")
		{
			m_streamSequenceOnce = m_configurationDictionary.get<bool>("streamSequenceOnce");
		}
	}

	void UltrasoundInterfaceBeamformedMock::configurationChanged()
	{
		readConfiguration();
	}

	bool UltrasoundInterfaceBeamformedMock::timerCallback() {
		if (!m_frozen)
		{
			double timestamp = getCurrentTime();

			m_callFrequency.measure();
			shared_ptr<USImage> pUSImage = std::make_shared<USImage>(*m_protoUSImage, m_pMockData);
			addData<0>(pUSImage);

			if (!m_singleImage)
			{
				if (m_lastFrame)
				{
					setRunning(false);
				}
				else
				{
					readNextFrame();
				}
			}
			m_callFrequency.measureEnd();
		}
		return getRunning();
	}

	void UltrasoundInterfaceBeamformedMock::readConfiguration()
	{
		lock_guard<mutex> lock(m_objectMutex);
		//read conf values
		m_singleImage = m_configurationDictionary.get<bool>("singleImage");
		m_streamSequenceOnce = m_configurationDictionary.get<bool>("streamSequenceOnce");
		m_frequency = m_configurationDictionary.get<double>("frequency");
		m_mockMetadataFilename = m_configurationDictionary.get<string>("mockMetaDataFilename");
		m_mockDataFilenames = split(m_configurationDictionary.get<string>("mockDataFilename"), ',');
		for (auto& filename : m_mockDataFilenames)
		{
			filename = trim(filename);
		}
	}

	void UltrasoundInterfaceBeamformedMock::readNextFrame()
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
				m_lastFrame = true;
			}
		}
	}

	bool UltrasoundInterfaceBeamformedMock::ready()
	{
		return true;
	}
}