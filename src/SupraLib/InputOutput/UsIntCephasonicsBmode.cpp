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
		return discoveredPlatforms[0];
	}

	UsIntCephasonicsBmode::UsIntCephasonicsBmode(tbb::flow::graph & graph, const std::string& nodeID)
		: AbstractInput<RecordObject>(graph, nodeID)
		, m_cPlatformHandle(nullptr)
		, m_cScanDefiniton(nullptr)
	{
		m_ready = false;

		if (!m_environSet)
		{
			setenv("CS_SYSCFG_SINGLE", "1", true);
			setenv("CS_LOGFILE", "", true);
			setenv("CS_BITFILE_FORCE", "1", true);
			m_environSet = true;
		}

		//Setup allowed values for parameters
		m_valueRangeDictionary.set<string>("xmlFileName", "", "xScan file");
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
		//Create execution thread to run USEngine
		m_runEngineThread = thread([this]() {
			//The run function of USEngine starts its internal state machine that will run infinitely
			//until the USEngine::teardown() function is called or a fatal exception.
			m_cUSEngine->run(*m_pDataProcessor);

			//This thread will only return null on teardown of USEngine.
		});

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
		static CallFrequency m("US");
		double timestamp = getCurrentTime();
		shared_ptr<USImage<uint8_t> > pImage;
		{
			lock_guard<mutex> lock(m_objectMutex);
			m.measure();

			size_t numVectors = frameBuffer->getNx();
			size_t numSamples = frameBuffer->getNy();

			shared_ptr<Container<uint8_t> > pData = make_shared<Container<uint8_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), numVectors*numSamples);
			std::memcpy(pData->get(), frameBuffer->getBuf(), numVectors*numSamples);

			pImage = make_shared<USImage<uint8_t> >(
				vec2s{ numVectors, numSamples }, pData, m_pImageProperties, timestamp, timestamp);
		}
		addData<0>(pImage);
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
