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

#include <memory>

#include <USPlatformMgr.h>
#include <PlatformHandle.h>
#include <ScanDef.h>
#include <USEngine.h>
#include <FrameBuffer.h>
#include <FrameDef.h>
#include <AddImageLayout.h>
#include <AddScanConverter.h>
#include <EnableScanConverter.h>
//#include <cmd/SetDecimFiltBypass.h>
#include <cmd/SetDecimFiltDecimation.h>
#include <cmd/SetDemodDecimFiltBypass.h>
#include <cmd/SetLogarithm.h>
#include <FCOS.h>

#include "UsIntCephasonicsBmodeProc.h"
#include "UsIntCephasonicsBmode.h"

#include <set>
#include <cstdlib>

#include "USImage.h"
#include "utilities/utility.h"
#include "utilities/CallFrequency.h"
#include "utilities/Logging.h"
#include "ContainerFactory.h"

using namespace std;
using namespace cs;
//TODO replace CS_THROW with something more intelligent

namespace supra
{
	using namespace logging;

	bool UsIntCephasonicsBmode::m_environSet = false;


	//Step 1 ------------------ "Setup Platform"
	PlatformHandle* UsIntCephasonicsBmode::setupPlatform()
	{
		PlatformHandle* discoveredPlatforms[USPlatformMgr::MAX_HANDLES];
		uint32 numPlatforms;

		//Returns the number of ultrasound platforms connected
		numPlatforms = USPlatformMgr::instance()->discoverPlatforms(discoveredPlatforms);

		//numPlatforms is out of range i.e. no platform connected
		if (numPlatforms < 1 || numPlatforms > USPlatformMgr::MAX_HANDLES)
			CS_THROW("Number of platforms is out of range, numPlatforms=" << numPlatforms);

		//if more than one platform is connected.
		if (numPlatforms > 1)
			CS_THROW("BasicBMode Demo supports only 1 platform connected, numPlatforms=" << numPlatforms);

		auto fcos = USPlatformMgr::getFCOSBits(*discoveredPlatforms[0]);
		
		logging::log_log("UsIntCephasonicsBmode: fcos rfmode: ", fcos.isSW_FCOS_rfmode_active());
	
		return discoveredPlatforms[0];
	}

	UsIntCephasonicsBmode::UsIntCephasonicsBmode(tbb::flow::graph & graph, const std::string& nodeID)
		: AbstractInput<RecordObject>(graph, nodeID)
		, m_cPlatformHandle(nullptr)
		, m_cScanDefiniton(nullptr)
		, m_processingStage(4)
		, m_rfStreaming(false)
	{
		m_ready = false;
		m_callFrequency.setName("US-CepBuiltin");

		if (!m_environSet)
		{
			setenv("CS_SYSCFG_SINGLE", "1", true);
			setenv("CS_LOGFILE", "", true);
			//setenv("CS_BITFILE_FORCE", "1", true);
			m_environSet = true;
		}

		//Setup allowed values for parameters
		m_valueRangeDictionary.set<string>("xmlFileName", "", "xScan file");
		m_valueRangeDictionary.set<int>("stage", 0, 4, 4, "Stage");
		m_valueRangeDictionary.set<bool>("rfStreaming", {true, false}, false, "RF Streaming");
	}

	UsIntCephasonicsBmode::~UsIntCephasonicsBmode()
	{
		//End of the world, waiting to join completed thread on teardown
		m_cUSEngine->tearDown();   //Teardown USEngine
		if (m_runEngineThread.joinable())
		{
			m_runEngineThread.join();
		}
	}

	void UsIntCephasonicsBmode::initializeDevice()
	{
		//Step 1 ------------------ "Setup Platform"
		m_cPlatformHandle = setupPlatform();
		m_cUSEngine = unique_ptr<USEngine>(new USEngine(*m_cPlatformHandle));
		m_cUSEngine->stop();
		m_cUSEngine->setBlocking(true);

		//Step 2 ----------------- "Create Scan Definition"
		m_cScanDefiniton = &ScanDef::createScanDef(*m_cPlatformHandle, m_xmlFileName);
		m_cUSEngine->setScanDef(*m_cScanDefiniton);

		//Step 3 ----------------- "Create Ultrasound Engine Thread"
		//create the data processor that later handles the data
		m_pDataProcessor = unique_ptr<UsIntCephasonicsBmodeProc>(
			new UsIntCephasonicsBmodeProc(*m_cPlatformHandle, this)
			);
		if (m_rfStreaming)
		{
			m_processingStage = 0;
		}
		m_pDataProcessor->setActiveStage(m_processingStage);

		//Create execution thread to run USEngine
		m_runEngineThread = thread([this]() {
			//The run function of USEngine starts its internal state machine that will run infinitely
			//until the USEngine::teardown() function is called or a fatal exception.
			m_cUSEngine->run(*m_pDataProcessor);

			//This thread will only return null on teardown of USEngine.
		});
		std::this_thread::sleep_for (std::chrono::seconds(2));
		
		if (m_rfStreaming)
		{
			m_cScanDefiniton->update(SetDecimFiltDecimation(2));
			m_cScanDefiniton->update(SetLogarithm(false));
			m_cScanDefiniton->update(SetDemodDecimFiltBypass(true));
		}

		logging::log_log("USEngine: initialized");

		m_ready = true;
	}

	void UsIntCephasonicsBmode::startAcquisition()
	{
		m_cUSEngine->start();
	}

	void UsIntCephasonicsBmode::stopAcquisition()
	{
		m_cUSEngine->stop();       //Stop USEngine
	}

	void UsIntCephasonicsBmode::configurationEntryChanged(const std::string & configKey)
	{
		if (configKey == "stage")
		{
			m_processingStage = m_configurationDictionary.get<int>("stage");
			if (m_ready)
			{
				m_pDataProcessor->setActiveStage(m_processingStage);
			}
		}
	}

	void UsIntCephasonicsBmode::configurationChanged()
	{
		readConfiguration();
	}

	void UsIntCephasonicsBmode::readConfiguration()
	{
		lock_guard<mutex> lock(m_objectMutex);
		//read conf values
		m_xmlFileName = m_configurationDictionary.get<string>("xmlFileName");
		m_processingStage = m_configurationDictionary.get<int>("stage");
		m_rfStreaming = m_configurationDictionary.get<bool>("rfStreaming");

		//TODO this is just to have somthing. Needs to be deduced / read from cephasonics
		//prepare the USImageProperties
		m_pImageProperties = make_shared<USImageProperties>(
			vec2s{ 128, 1 },
			512,
			USImageProperties::ImageType::BMode,
			USImageProperties::ImageState::Scan,
			USImageProperties::TransducerType::Linear,
			60);
	}

	void UsIntCephasonicsBmode::layoutChanged(ImageLayout & layout)
	{
		//Update USImageProperties from new image Layout
		if (layout.getFrameIDs().size() != 1)
		{
			log_error("Multiple frame sequences are not supported yet.");
		}
		else {
			set<uint16_t> frameIDs = layout.getFrameIDs();
			uint16_t frameID = *(frameIDs.begin());

			m_pImageProperties = make_shared<USImageProperties>(
				vec2s{ layout.getFrameWidth(frameID), 1 },
				layout.getFrameHeight(frameID),
				USImageProperties::ImageType::BMode,
				USImageProperties::ImageState::Scan,
				USImageProperties::TransducerType::Linear,
				60);
		}
	}

	void UsIntCephasonicsBmode::putData(FrameBuffer * frameBuffer)
	{
		double timestamp = getCurrentTime();
		shared_ptr<USImage<uint8_t> > pImage;
		{
			lock_guard<mutex> lock(m_objectMutex);
			m_callFrequency.measure();

			size_t numVectors = frameBuffer->getNx();
			size_t numSamples = frameBuffer->getNy();

			shared_ptr<Container<uint8_t> > pData = make_shared<Container<uint8_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), numVectors*numSamples);
			std::memcpy(pData->get(), frameBuffer->getBuf(), numVectors*numSamples);

			pImage = make_shared<USImage<uint8_t> >(
				vec2s{ numVectors, numSamples }, pData, m_pImageProperties, timestamp, timestamp);
		}
		addData<0>(pImage);
	}

	void UsIntCephasonicsBmode::putData(ScanData * scanData)
	{
		double timestamp = getCurrentTime();
		{
			lock_guard<mutex> lock(m_objectMutex);
			m_callFrequency.measure();

			size_t numVectors = scanData->numBeams;
			size_t numSamples = scanData->samplesPerBeam;

			switch(scanData->bytesPerSample)
			{
				case 1:
				{
					shared_ptr<USImage<uint8_t> > pImage;
					shared_ptr<Container<uint8_t> > pData = make_shared<Container<uint8_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), numVectors*numSamples);
					memcpyTransposed(pData->get(), reinterpret_cast<uint8_t*>(scanData->data), numVectors, numSamples);

					pImage = make_shared<USImage<uint8_t> >(
						vec2s{ numVectors, numSamples }, pData, m_pImageProperties, timestamp, timestamp);
					addData<0>(pImage);
					break;
				}
				case 2: 
				{
					shared_ptr<USImage<int16_t> > pImage;
					shared_ptr<Container<int16_t> > pData = make_shared<Container<int16_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), numVectors*numSamples);
					memcpyTransposed(pData->get(), reinterpret_cast<int16_t*>(scanData->data), numVectors, numSamples);

					pImage = make_shared<USImage<int16_t> >(
						vec2s{ numVectors, numSamples }, pData, m_pImageProperties, timestamp, timestamp);
					addData<0>(pImage);
					break;
				}
				case 4:
				{
					shared_ptr<USImage<int16_t> > pImage;
					shared_ptr<Container<int16_t> > pData = make_shared<Container<int16_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), numVectors*numSamples);
					int32_t* inData = reinterpret_cast<int32_t*>(scanData->data);

					for (size_t sample = 0; sample < numSamples; sample++)
					{
						for (size_t scanline = 0; scanline < numVectors; scanline++)
						{
							pData->get()[scanline + sample*numVectors] = static_cast<int16_t>(
								min(max(inData[sample + scanline*numSamples] >> 8,
									(int32_t)std::numeric_limits<int16_t>::min()
								), (int32_t)std::numeric_limits<int16_t>::max()
							));
						}
					}
					pImage = make_shared<USImage<int16_t> >(
						vec2s{ numVectors, numSamples }, pData, m_pImageProperties, timestamp, timestamp);
					addData<0>(pImage);
					break;
				}
				case 8:
					logging::log_error("UsIntCephasonicsBmode::putData: IQ data not handled yet");
					break;
				default:
					logging::log_error("UsIntCephasonicsBmode::putData: unhandled datatype");
					break;
			}
		}
		
	}

	bool UsIntCephasonicsBmode::ready()
	{
		return m_ready;
	}

	void UsIntCephasonicsBmode::freeze()
	{
		m_cUSEngine->stop();
	}

	void UsIntCephasonicsBmode::unfreeze()
	{
		m_cUSEngine->start();
	}
}
