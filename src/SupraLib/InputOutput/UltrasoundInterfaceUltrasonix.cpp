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


#include "UltrasoundInterfaceUltrasonix.h"

#include <Utilities/utility.h>
#include <utilities/Logging.h>
#include <utilities/CallFrequency.h>
#include <ContainerFactory.h>

#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace supra::logging;

#define MACROgetLastError getLastError(__FILE__, __LINE__)

namespace supra
{
	UltrasoundInterfaceUltrasonix* g_pUSLib = 0;

	UltrasoundInterfaceUltrasonix::UltrasoundInterfaceUltrasonix(tbb::flow::graph& graph, const std::string& nodeID)
		: AbstractInput(graph, nodeID, 2)
		, m_tickTimestamp(0)
		, m_imagingMode(usModeBmode)
		, m_imagingFormat(USImagingFormat{false, false, false})
		, m_frozen(false)
		, m_dataMask(0)
		, m_colorEnsembleSize(0)
	{
		m_callFrequency.setName("Ultrasonix");

		g_pUSLib = this;
		m_pUlterius = new ulterius();

		m_initialized = false;

		m_protoImageProperties = make_shared<USImageProperties>(
			vec2s{ 128, 1 },
			512,
			USImageProperties::ImageType::BMode,
			USImageProperties::ImageState::RF,
			USImageProperties::TransducerType::Linear,
			60);

		//TODO figure out value ranges
		//Preload Value Range Dictionary with ranges of allowed values
		m_valueRangeDictionary.set<string>("remoteIP", "127.0.0.1", "Ultrasonix IP");
		m_valueRangeDictionary.set<string>("imagingMode", { "BMode", "BMode + Color", "BMode + MMode", "BMode + Pulse" }, "BMode", "Imaging");
		m_valueRangeDictionary.set<bool>("bmodeRF", { false, true }, false, "Stream BMode RF");
		m_valueRangeDictionary.set<bool>("colordopplerRF", { false, true }, false, "Stream Color RF");
		m_valueRangeDictionary.set<bool>("pulsedopplerRF", { false, true }, false, "Stream Pulse RF");
		m_valueRangeDictionary.set<double>("speedOfSound", 1000, 3000, 1540, "Speed of sound [m/s]");
	}

	UltrasoundInterfaceUltrasonix::~UltrasoundInterfaceUltrasonix()
	{
		if (m_pUlterius) {
			m_pUlterius->disconnect();
			delete m_pUlterius;
		}
	}

	void UltrasoundInterfaceUltrasonix::initializeDevice()
	{
		// Connect to ulterius
		bool initSuccess = true;
		initSuccess &= m_pUlterius->connect((char*)m_remoteIp.c_str());
		if (!initSuccess) {
			log_error("Could not connect to Ultrasonix.");
		}
		//initSuccess &= m_pUlterius->setSharedMemoryStatus(0);

		setImaging();

		//enum uVariableType
		//{
		//	uTypeInteger = 0,
		//	uTypeFloat = 1,
		//	uTypeString = 2,
		//	uTypeGainCurve = 3,
		//	uTypeRectangle = 4,
		//	uTypeCurve = 5,
		//	uTypeColor = 6,
		//	uTypeBoolean = 7,
		//	uTypeDontKnow = 8,
		//	uTypePoint = 9
		//};
		//fetching all the parameters
		//uParam parameterStruct;
		//int parameterIndex = 0;
		//while (m_pUlterius->getParam(parameterIndex, parameterStruct))
		//{
		//	cout << "US Parameter: #" << parameterIndex << ", " << parameterStruct.id << ", " << parameterStruct.name << ", " << parameterStruct.source << ", " << parameterStruct.type << ", " << parameterStruct.unit << endl;
		//	parameterIndex++;
		//}


		m_initialized = initSuccess;

		// Initialize the parameters from the US machine itself
		initializeParametersFromDevice();
	}

	bool UltrasoundInterfaceUltrasonix::ready()
	{
		return m_initialized;
	}

	void UltrasoundInterfaceUltrasonix::freeze()
	{
		// if freeze has not been set before
		bool expected = false;
		if (m_frozen.compare_exchange_strong(expected, true))
		{
			toggleFreeze();
		}
	}

	void UltrasoundInterfaceUltrasonix::unfreeze()
	{
		// if freeze has been set before
		bool expected = true;
		if (m_frozen.compare_exchange_strong(expected, false))
		{
			toggleFreeze();
		}
	}

	void UltrasoundInterfaceUltrasonix::startAcquisition()
	{
		m_pUlterius->setCallback(UltrasoundInterfaceUltrasonix::callbackHandler);
		m_pUlterius->setParamCallback(UltrasoundInterfaceUltrasonix::callbackParamHandler);

		// ensure imaging is running
		if (m_pUlterius->getFreezeState() == 1) {
			m_pUlterius->toggleFreeze();
		}

		manageParameterUpdates();
	}

	void UltrasoundInterfaceUltrasonix::stopAcquisition()
	{
		lock_guard<mutex> lock(m_objectMutex);
		// stop imaging
		if (m_pUlterius->getFreezeState() == 0) {
			m_pUlterius->toggleFreeze();
		}
	}

	void UltrasoundInterfaceUltrasonix::configurationEntryChanged(const std::string & configKey)
	{
		lock_guard<mutex> lock(m_objectMutex);
		if (configKey == "imagingMode" || 
			configKey == "bmodeRF" ||
			configKey == "colordopplerRF" ||
			configKey == "pulsedopplerRF")
		{
			readImagingConfiguration();
			setImaging();
		}
	}

	void UltrasoundInterfaceUltrasonix::configurationChanged()
	{
		m_remoteIp = m_configurationDictionary.get<string>("remoteIP");
		m_speedOfSound = m_configurationDictionary.get<double>("speedOfSound");
		readImagingConfiguration();
	}

	void UltrasoundInterfaceUltrasonix::readImagingConfiguration()
	{		

		std::string imagingString = m_configurationDictionary.get<string>("imagingMode");
		if (imagingString == "BMode")
		{
			m_imagingMode = usModeBmode;
		}
		else if (imagingString == "BMode + Color")
		{
			m_imagingMode = usModeBmodeAndColordoppler;
		}
		else if (imagingString == "BMode + MMode")
		{
			m_imagingMode = usModeBmodeAndMmode;
		}
		else if (imagingString == "BMode + Pulse")
		{
			m_imagingMode = usModeBmodeAndPulseddoppler;
		}

		m_imagingFormat.bmodeRF = m_configurationDictionary.get<bool>("bmodeRF");
		m_imagingFormat.colordopplerRF = m_configurationDictionary.get<bool>("colordopplerRF");
		m_imagingFormat.pulsedopplerRF = m_configurationDictionary.get<bool>("pulsedopplerRF");
	}

	void UltrasoundInterfaceUltrasonix::manageParameterUpdates()
	{
		while (getRunning()) {
			std::unique_lock<std::mutex> lock(m_parameterMutex);
			m_cvParameterUpdates.wait(lock, [this] { return !m_pParameterQueue.empty(); });
			while (!m_pParameterQueue.empty()) {
				shared_ptr<string> pParamName;
				bool haveParam = false;
				{
					lock_guard<mutex> queuelock(m_queueMutex);
					if (!m_pParameterQueue.empty())
					{
						pParamName = m_pParameterQueue.front();
						m_pParameterQueue.pop();
						haveParam = true;
					}
				}
				if (haveParam)
				{
					ulteriusParamChanged(pParamName);
				}
			}
		}
	}

	template <typename ImageType>
	void UltrasoundInterfaceUltrasonix::copyAndSendNewImage(size_t port, std::shared_ptr<UlteriusPacket> packet,
		std::shared_ptr<const USImageProperties> imProp)
	{
		// copy the data into a buffer managed by us (i.e. a shared pointer)
		// and create the image
		auto spData = make_shared<Container<ImageType> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(),
			imProp->getNumSamples()*imProp->getNumScanlines()*imProp->getNumChannels());
		memcpyTransposed(spData->get(), (ImageType*)(packet->pData), imProp->getNumScanlines()*imProp->getNumChannels(), imProp->getNumSamples());
		shared_ptr<USImage<ImageType> > pImage = make_shared < USImage<ImageType> >(
			vec3s{ imProp->getNumScanlines(), imProp->getNumSamples(), 1 }, spData, imProp, packet->dTimestamp, packet->dTimestamp, imProp->getNumChannels());

		addData(port, pImage);
	}

	template <typename ImageType>
	void UltrasoundInterfaceUltrasonix::copyAndSendNewColorImage(size_t port, std::shared_ptr<UlteriusPacket> packet,
		std::shared_ptr<const USImageProperties> imProp)
	{
		// copy the data into a buffer managed by us (i.e. a shared pointer)
		// and create the image
		auto spData = make_shared<Container<ImageType> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(),
			imProp->getNumSamples()*imProp->getNumScanlines()*imProp->getNumChannels());
		memcpy(spData->get(), (ImageType*)(packet->pData), 
			imProp->getNumScanlines() * imProp->getNumSamples() * imProp->getNumChannels());
		shared_ptr<USImage<ImageType> > pImage = make_shared < USImage<ImageType> >(
			vec3s{ imProp->getNumScanlines(), imProp->getNumSamples(), 1 }, spData, imProp, packet->dTimestamp, packet->dTimestamp, imProp->getNumChannels());

		addData(port, pImage);
	}

	/// Processes received Ulterius packet and emits it into the graph
	bool UltrasoundInterfaceUltrasonix::processUlteriusPacket(shared_ptr<UlteriusPacket> packet) {

		//std::cout << " frame : " << packet->iFramenum << ", rate:  " << 1.0/(packet->dTimestamp-m_tickTimestamp) << ", ";
		m_tickTimestamp = packet->dTimestamp;
		{
			lock_guard<mutex> lock(m_objectMutex);

			// Check whether the data type that arrived is within those we are interested in
			if (m_dataMask & packet->iType)
			{
				int port = m_dataToOutputMap[packet->iType].first;
				auto imProp = m_dataToOutputMap[packet->iType].second;

				int sampleSize = 1; // Bytes per sample
				size_t channels = 1;
				if (packet->iType == udtBPre ||
					packet->iType == udtMPre ||
					packet->iType == udtMPost ||
					packet->iType == udtPWSpectrum)
				{
					sampleSize = 1; // Bytes
				}
				else if (packet->iType == udtRF ||
					packet->iType == udtPWRF)
				{
					sampleSize = 2; // Bytes
				}
				else if (packet->iType == udtColorRF)
				{
					sampleSize = 2; // Bytes
					channels = m_colorEnsembleSize;
				}
				else if (packet->iType == udtColorCombined)
				{
					sampleSize = 1; // Bytes
					channels = 4;
				}

				if (packet->iSize != imProp->getNumSamples() * imProp->getNumScanlines() * channels * sampleSize)
				{
					log_log("Ulterius packet did not have expected size. Was: ", packet->iSize, " bytes, expected ", 
						imProp->getNumSamples() * imProp->getNumScanlines() * channels * sampleSize, 
						" = ", imProp->getNumScanlines(), " * ", imProp->getNumSamples(), " * ", channels, " * ", sampleSize);
					log_warn("Dropping incomplete frame");
					updateImagingParams();
					return false;
				}

				if (packet->iType == udtBPre ||
					packet->iType == udtMPre ||
					packet->iType == udtMPost ||
					packet->iType == udtPWSpectrum)
				{
					copyAndSendNewImage<uint8_t>(port, packet, imProp);
				}
				else if (packet->iType == udtRF ||
					packet->iType == udtPWRF ||
					packet->iType == udtColorRF)
				{
					copyAndSendNewImage<int16_t>(port, packet, imProp);
				}
				else if (packet->iType == udtColorCombined)
				{
					copyAndSendNewColorImage<uint8_t>(port, packet, imProp);
				}

				if (port == 0)
				{
					m_callFrequency.measure();
				}
			}
		}
		return true;
	}


	bool UltrasoundInterfaceUltrasonix::callbackHandler(void * data, int type, int sz, bool cine, int frmnum) {

		if (g_pUSLib) {

			double timestamp = getCurrentTime();

			shared_ptr<UlteriusPacket> packet = make_shared<UlteriusPacket>();
			packet->bCine = cine;
			packet->iType = type;
			packet->iSize = sz;
			packet->iFramenum = frmnum;
			packet->dTimestamp = timestamp;

			if (data != 0 && packet->iSize != 0)
			{
				packet->pData = data;

				//Needs to be synchronous!
				g_pUSLib->processUlteriusPacket(packet);
			}
			else
			{
				return false;
			}
		}

		return true;
	}


	bool UltrasoundInterfaceUltrasonix::ulteriusParamCallback(void* paramID, int ptX, int ptY) {
		shared_ptr<string> paramName = make_shared<string>((char*)paramID);
		{
			lock_guard<mutex> queueLock(m_queueMutex);
			m_pParameterQueue.push(paramName);
		}
		m_cvParameterUpdates.notify_one();
		return true;
	}

	void UltrasoundInterfaceUltrasonix::ulteriusParamChanged(shared_ptr<string> pParamName) {
		string paramName(*pParamName);
		lock_guard<mutex> lock(m_objectMutex);

		//prepare new USImageProperties
		vector<shared_ptr<USImageProperties> > imProps;
		for (auto pair : m_dataToOutputMap)
		{
			imProps.push_back(make_shared<USImageProperties>(*pair.second.second));
		}
		

		bool paramOK = true;
		bool needUpdateOfImagingParams = false;
		if (paramName == "mode id")
		{
			// imaging modes (Ulterius knows about)
			// bmode only				= 0
			// bmode + m mode			= 1
			// bmode + color			    = 2
			// bmode + Pulsed doppler	= 3
			// bmode + Bmode RF			= 12 (somewhere I also read 16 for that)
			
			int ulteriusImagingMode = m_pUlterius->getActiveImagingMode();
			log_log("Ultrasonix: Imaging-Mode: ", ulteriusImagingMode);

			// Figure out which of USImagingMode is currently selected on the machine
			USImagingMode currentModeOnMachine = usModeBmode;
			switch (ulteriusImagingMode)
			{
				case 0: // bmode only
				{
					currentModeOnMachine = usModeBmode;
					break;
				}
				case 1: // bmode + m mode
				{
					currentModeOnMachine = usModeBmodeAndMmode;
					break;
				}
				case 2: // bmode + color
				{
					currentModeOnMachine = usModeBmodeAndColordoppler;
					break;
				}
				case 3: // bmode + Pulsed doppler
				{
					currentModeOnMachine = usModeBmodeAndPulseddoppler;
					break;
				}
				case 12: // bmode + Bmode RF
				{
					currentModeOnMachine = usModeBmode;
					break;
				}
				default:
				{
					log_error("Ultrasonix: Error: unsupported mode! please update mode ", ulteriusImagingMode);
					break;
				}
			}

			if (currentModeOnMachine != m_imagingMode)
			{
				// The imaging mode was changed for some reason, but we did not trigger this.
				// Try to convince Ultrasonix to please get back to the mode we want
				setImaging(); 
			}
		}
		else if (paramName == "probe id")
		{
			char* tmp = new char[80];
			paramOK &= m_pUlterius->getActiveProbe(tmp, 80);
			if (strncmp(tmp, "C5-2/60", 80) == 0)
			{
				for (auto & imProp : imProps)
				{
					imProp->setTransducerType(USImageProperties::Curved);
					imProp->setScanlineLayout({ 128, 1 });
					//imProp->setProbeRadius(60);
					//imProp->setFov(degToRad(60.0));
				}
				needUpdateOfImagingParams = true;
			}
			else if (strncmp(tmp, "L14-5/38", 80) == 0)
			{
				for (auto & imProp : imProps)
				{
					imProp->setTransducerType(USImageProperties::Linear);
					imProp->setScanlineLayout({ 128, 1 });
					//imProp->setProbeRadius(0);
					//imProp->setFov(0.3*(128 - 1));
				}
				needUpdateOfImagingParams = true;
			}
			log_log("changed probe id: ", tmp);
		}
		else if (paramName == "freezestatus")
		{
			log_log("changed freeze status");
		}
		else if (paramName == "b-depth")
		{
			int value;
			needUpdateOfImagingParams = true;
			paramOK &= m_pUlterius->getParamValue("b-depth", value);
			for (auto & imProp : imProps)
			{
				imProp->setDepth(static_cast<double>(value));
			}
			log_log("changed b-depth: ", value);
		}
		else if (paramName == "b-ldensity")
		{
			int value;
			needUpdateOfImagingParams = true;
			paramOK &= m_pUlterius->getParamValue("b-ldensity", value);
			log_log("changed b-ldensity: ", value);
		}
		else if (paramName == "b-tgc")
		{
			uTGC tgcValue;
			paramOK &= m_pUlterius->getParamValue("b-tgc", tgcValue);
			vector<int> tgcVector;
			tgcVector.push_back(tgcValue.v1);
			tgcVector.push_back(tgcValue.v2);
			tgcVector.push_back(tgcValue.v3);
			tgcVector.push_back(tgcValue.v4);
			tgcVector.push_back(tgcValue.v5);
			tgcVector.push_back(tgcValue.v6);
			tgcVector.push_back(tgcValue.v7);
			tgcVector.push_back(tgcValue.v8);
			vector<double> tgcDepthVector;
			for (int k = 0; k < 8; ++k) {
				tgcDepthVector.push_back((static_cast<double>(k) + 0.5)*100.0 / 8.0);
			}
			for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter("b-tgc", tgcVector);
				imProp->setSpecificParameter("b-tgc relative depth", tgcDepthVector);
			}
			log_log("changed b-tgc");
		}
		else if (paramName == "rf-mode")
		{
			int value;
		    paramOK &= m_pUlterius->getParamValue("rf-mode", value);
			log_log("Ultrasonix: rf-mode changed: ", value);
		}
		else if (paramName == "color-gain") 
		{
			int value;
			paramOK &= m_pUlterius->getParamValue("color-gain", value);
			for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter("color-gain", value);
			}
			log_log("changed color-gain: ", value);
		}
		else if (paramName == "color-ensemble") 
		{
			int value;
		    paramOK &= m_pUlterius->getParamValue("color-ensemble", value);
			m_colorEnsembleSize = value;
		    for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter("color-ensemble", value);
			}
			log_log("changed color-ensemble: ", value);
		}
		else if (paramName == "color-numDummyLines") 
		{
			int value;
		    paramOK &= m_pUlterius->getParamValue("color-numDummyLines", value);
		    for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter("color-numDummyLinesc", value);
			}
			log_log("changed color-numDummyLines: ", value);
		}
		else if (paramName == "color-prp") 
		{
			int value;
		    paramOK &= m_pUlterius->getParamValue("color-prp", value);
		    //m_pCurrentContextMap->m_cfmPRFInHz				= 1000.0/(static_cast<double>(value)/1000);
		    for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter("color-prpc", value);
			}
			log_log("changed color-prp: ", value);
		}
		else if (paramName == "color-freq") 
		{
			int value;
		    paramOK &= m_pUlterius->getParamValue("color-freq", value);
		    //m_pCurrentContextMap->m_cfmRxFrequInHz			= static_cast<double>(value);
		    for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter("color-freq", value);
			}
			log_log("changed color-freq: ", value);
		} 
		else if (paramName == "color-deviation")
		{
			int value;
		    paramOK &= m_pUlterius->getParamValue("color-deviation", value);
		    //m_pCurrentContextMap->m_cfmSteeringAngleInRad = M_PI*180.0 / (static_cast<double>(value)/1000.0);	
			for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter("color-deviation", value);
			}
			log_log("changed color-deviation: ", value);
		}
		//  else if (paramName == "color-ldensity")
		//  {
		//      // update both color and bmode density, as this may change for doppler acquisition
		//      // und ultrasonix unfortunately uses separate density modes for bmode in bmode imaging
		//      // and bmode in color imaging

		//      int clineDensity = 0;
		//      paramOK &= m_pUlterius->getParamValue("color-ldensity", clineDensity);

		//      int lDensity = 0;
		//      paramOK &= m_pUlterius->getParamValue("b-ldensity", lDensity);

		//      m_pCurrentContextMap->m_bmodeVectorsPerMOrRad = static_cast<double>(lDensity) / (m_pCurrentContextMap->m_probePitchInM * m_pCurrentContextMap->m_probeNumElements);
		//      m_pCurrentContextMap->m_cfmVectorsPerMOrRad = static_cast<double>(clineDensity) / (m_pCurrentContextMap->m_probePitchInM * m_pCurrentContextMap->m_probeNumElements);
			  //std::cout << "changed b-ldensity: " << m_pCurrentContextMap->m_bmodeVectorsPerMOrRad;
			  //std::cout << ", cfm-ldensity: " << m_pCurrentContextMap->m_cfmVectorsPerMOrRad << std::endl;

		//  }
		//  else if (paramName == "color-color rect")
		//  { 
		//      uRect colorRectangle;
		//      paramOK &= m_pUlterius->getParamValue("color-color rect", colorRectangle);

		//      int downSamplingFrequ = 0;
		//      paramOK &= m_pUlterius->getParamValue("color-csampl freq", downSamplingFrequ);

		//      double startSample = colorRectangle.top * m_pCurrentContextMap->m_cfmSamplingFrequInHz/static_cast<double>(downSamplingFrequ);
		//      double endSample = colorRectangle.bottom * m_pCurrentContextMap->m_cfmSamplingFrequInHz/static_cast<double>(downSamplingFrequ);

		//      double startVector = colorRectangle.left;
		//      double endVector = colorRectangle.right;

		//      m_pCurrentContextMap->m_cfmStartSample = startSample;
		//      m_pCurrentContextMap->m_cfmStartVector = startVector;

		//      std::cout << "color-color rect changed: " << startVector << ", " << startSample << "; " << m_pCurrentContextMap->m_cfmNumVectors << ", " << m_pCurrentContextMap->m_cfmNumSamples << std::endl;
		//  }
		//  else if (paramName == "color-rf decimation")
		//  {
		//      paramOK &= m_pUlterius->getParamValue("color-rf decimation", value);

		//      m_pCurrentContextMap->m_cfmSamplingFrequInHz = 40000000;
		//      for (int k=0; k<value; ++k)
		//      {
		//          m_pCurrentContextMap->m_cfmSamplingFrequInHz *= 0.5;
		//      }

		//      std::cout << "color-sampling frequency changed: " << m_pCurrentContextMap->m_cfmSamplingFrequInHz << std::endl;
		//  }
		else if (paramName == "rf-rf decimation")
		{
			int value;
		    paramOK &= m_pUlterius->getParamValue("rf-rf decimation", value);
			
			m_rfSamplingFrequInHz = 40000000;
		    for (int k=0; k<value; ++k)
		    {
		        m_rfSamplingFrequInHz *= 0.5;
		    }
		    log_log("Ultrasonix: rf-sampling frequency changed: ", m_rfSamplingFrequInHz);
		}
		else if (
			paramName == "frame rate" ||
			paramName == "b-freq" ||
			paramName == "b-sampl freq" ||
			paramName == "b-deviation" ||
			paramName == "b-gain" ||
			paramName == "power" ||
			paramName == "b-focus count" ||
			paramName == "focus depth" ||
			//TODO THIS IS JUST A TEST
			paramName == "microns" ||
			paramName == "frame period" ||
			paramName == "b frame rate" ||
			paramName == "b-hd-density" ||
			paramName == "ldensity adjust")

		{
			int value;
			paramOK &= m_pUlterius->getParamValue(paramName.c_str(), value);
			for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter(paramName, value);
			}
			log_log("changed ", paramName, ": ", value);
		}
		else if (
			paramName == "b-image rect" ||
			paramName == "b-hd-image rect")
		{
			uRect rect;
			paramOK &= m_pUlterius->getParamValue(paramName.c_str(), rect);
			string valueStr("[" + to_string(rect.left) + ", " + to_string(rect.top) + ", " + to_string(rect.right) + ", " + to_string(rect.bottom) + "]");
			for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter<string>(paramName, valueStr);
			}
			log_log("changed ", paramName, ": ", valueStr);
		}
		else if (
			paramName == "origin")
		{
			uPoint point;
			paramOK &= m_pUlterius->getParamValue(paramName.c_str(), point);
			for (auto & imProp : imProps)
			{
				imProp->setSpecificParameter<string>(paramName, "[" + to_string(point.x) + ", " + to_string(point.y) + "]");
			}
		}
		else
		{
			log_info("unhandled parameter: ", paramName);
		}

		if (!paramOK)
		{
			log_error("Error updating parameter : ", paramName);
			MACROgetLastError;
		}

		size_t i = 0;
		for (auto& pair : m_dataToOutputMap)
		{
			pair.second.second = imProps[i++];
		}

		if (needUpdateOfImagingParams)
		{
			updateImagingParams();
		}
	}


	bool UltrasoundInterfaceUltrasonix::callbackParamHandler(void* paramID, int ptX, int ptY) {
		if (g_pUSLib) {
			g_pUSLib->ulteriusParamCallback(paramID, ptX, ptY);
		}
		return true;
	}


	void UltrasoundInterfaceUltrasonix::toggleFreeze()
	{
		m_pUlterius->toggleFreeze();
	}

	/// The imaging mode defines which types of acquisitions are performed on the machine
	void UltrasoundInterfaceUltrasonix::setImaging()
	{
		// imaging modes (Ulterius knows about)
		// bmode only				= 0
		// bmode + m mode			= 1
		// bmode + color			    = 2
		// bmode + Pulsed doppler	= 3
		// bmode + Bmode RF			= 12 (somewhere I also read 16 for that)

		//HERE
	    int newUlteriusImageMode = 0;
		switch (m_imagingMode)
		{
			case usModeBmode:
			{
				if (m_imagingFormat.bmodeRF)
				{
					newUlteriusImageMode = 12;
				}
				else
				{
					newUlteriusImageMode = 0;
				}
				break;
			}
			case usModeBmodeAndColordoppler:
			{
				newUlteriusImageMode = 2;
				break;
			}
			case usModeBmodeAndMmode:
			{
				newUlteriusImageMode = 1;
				break;
			}
			case usModeBmodeAndPulseddoppler:
			{
				newUlteriusImageMode = 3;
				break;
			}
			default:
			{
				log_error("Ultrasonix: Unhandled UsImagingMode.");
				break;
			}
		}
		
		if (!m_pUlterius->selectMode(newUlteriusImageMode))
		{
			log_error("Ultrasonix: Failed to set imaging mode ", newUlteriusImageMode);
			MACROgetLastError;
		}
	
	    updateDataStreaming();
	}

	/// Updates settings on the machine: Which data to stream and in which format.
	/// The data that can be streamed depends on the selected imaging mode
	void UltrasoundInterfaceUltrasonix::updateDataStreaming() {
		m_dataMask = 0;
		m_dataToOutputMap.clear();
		int outputsUsed = 0;

		if (m_imagingFormat.bmodeRF || m_imagingFormat.colordopplerRF || m_imagingFormat.pulsedopplerRF)
		{
			m_pUlterius->setParamValue("rf-mode", 2);
		}
		else
		{
			m_pUlterius->setParamValue("rf-mode", 0);
		}

		if (m_imagingFormat.bmodeRF)
		{
			m_dataMask |= udtRF;
			m_dataToOutputMap[udtRF] = std::make_pair(outputsUsed++, m_protoImageProperties);
		}
		else
		{
			m_dataMask |= udtBPre;
			m_dataToOutputMap[udtBPre] = std::make_pair(outputsUsed++, m_protoImageProperties);
		}

		if (m_imagingMode == usModeBmodeAndColordoppler)
		{
			if (m_imagingFormat.colordopplerRF)
			{
				//m_pUlterius->setParamValue("color-rf decimation", 2); // To not loose frames? I suppose
				m_dataMask |= udtColorRF;
				m_dataToOutputMap[udtColorRF] = std::make_pair(outputsUsed++, m_protoImageProperties);
			}
			else
			{
				m_dataMask |= udtColorCombined;
				m_dataToOutputMap[udtColorCombined] = std::make_pair(outputsUsed++, m_protoImageProperties);
			}
		}
		if (m_imagingMode == usModeBmodeAndMmode)
		{
			m_dataMask |= udtMPre;
			m_dataToOutputMap[udtMPre] = std::make_pair(outputsUsed++, m_protoImageProperties);
		}
		if (m_imagingMode == usModeBmodeAndPulseddoppler)
		{
			if (m_imagingFormat.colordopplerRF)
			{
				m_dataMask |= udtPWRF;
				m_dataToOutputMap[udtPWRF] = std::make_pair(outputsUsed++, m_protoImageProperties);
			}
			else
			{
				m_dataMask |= udtPWSpectrum;
				m_dataToOutputMap[udtPWSpectrum] = std::make_pair(outputsUsed++, m_protoImageProperties);
			}
		}

		//if (usModeBMode == mode) {
		//	// switch to BMode
		//	dataMask = udtBPre;
		//	m_pUlterius->setParamValue("rf-mode", 0);
		//}
		//else if(mode == usModeBModeRF)
		//{
		//	dataMask = udtRF | udtBPre;
		//	
		//} 
		//else if (usModeCFM == mode) {
		//    
	 //       
		//	//HERE
	 //       /*if (rfEnabled)
	 //       {
	 //           std::cout << "Warning: simultaneous RF and ColorRF will result in massive dropped frames for CRF" << std::endl;
	 //       }*/
	
		//	// switch to Color Doppler
		//	//dataMask = rfEnabled ? (udtColorRF | udtRF) : (udtColorRF | udtBPre); // works only from 6.07 onwards!
	 //       //dataMask = udtColorRF;
		//}
	
	    // update
		m_pUlterius->setDataToAcquire(m_dataMask);
	}

	void UltrasoundInterfaceUltrasonix::getLastError(const char* file, int line)
	{
		char errBuf[512];
		bool errMessageOK = m_pUlterius->getLastError(errBuf, 512);
		if (errMessageOK)
		{
			log_error("The error was: '", errBuf, "'");
		}
		else {
			log_error("additionally, the error message could not be retrieved");
		}
		if (file)
		{
			log_error("Location: ", file, ":", line);
		}
	}

	/// Get the details of transmitted images (size, etc.) from the machine
	void UltrasoundInterfaceUltrasonix::updateImagingParams()
	{
		for (auto & dataToOutput : m_dataToOutputMap)
		{
			int dataType = get<0>(dataToOutput);
			
			uDataDesc dataDescriptor;
			if (m_pUlterius->getDataDescriptor((uData)dataType, dataDescriptor))
			{
				int numVectors = dataDescriptor.w;
				int numSamples = dataDescriptor.h;

				shared_ptr<USImageProperties> imProp = make_shared<USImageProperties>(*(dataToOutput.second.second));
				imProp->setNumSamples(numSamples);
				imProp->setScanlineLayout({ (size_t)numVectors, 1 });
				if (dataType == udtColorCombined)
				{
					imProp->setImageState(USImageProperties::ImageState::Scan);
				}
				if (dataType == udtBPre || dataType == udtPWSpectrum)
				{
					imProp->setImageState(USImageProperties::ImageState::PreScan);
				}
				else if (dataType == udtRF || dataType == udtColorRF || dataType == udtPWRF)
				{
					imProp->setImageState(USImageProperties::ImageState::RF);
				}

				if (dataType == udtBPre || dataType == udtRF)
				{
					imProp->setImageType(USImageProperties::ImageType::BMode);
				}
				else if (dataType == udtPWSpectrum || dataType == udtColorRF || dataType == udtPWRF || dataType == udtColorCombined)
				{
					imProp->setImageType(USImageProperties::ImageType::Doppler);
					if (dataType == udtColorCombined)
					{
						imProp->setNumChannels(4);
					}
					if (dataType == udtColorRF)
					{
						imProp->setNumChannels(m_colorEnsembleSize);
					}
				}

				dataToOutput.second.second = imProp;

				log_info("Ultrasonix: New image params for data type ", dataType, " - vectors : ", numVectors, ", samples : ", numSamples);
			}
			else
			{
				log_error("Error fetching data descriptor for data type ", dataType);
				MACROgetLastError;
			}
		}
	}


	void UltrasoundInterfaceUltrasonix::initializeParametersFromDevice()
	{
		//do updates on all relevant parameters
		ulteriusParamChanged(make_shared<string>("probe id"));
		ulteriusParamChanged(make_shared<string>("mode id"));
		ulteriusParamChanged(make_shared<string>("b-depth"));
		ulteriusParamChanged(make_shared<string>("b-ldensity"));
		ulteriusParamChanged(make_shared<string>("frame rate"));
		ulteriusParamChanged(make_shared<string>("b-freq"));
		ulteriusParamChanged(make_shared<string>("b-sampl freq"));
		ulteriusParamChanged(make_shared<string>("b-deviation"));
		ulteriusParamChanged(make_shared<string>("b-gain"));
		ulteriusParamChanged(make_shared<string>("power"));
		ulteriusParamChanged(make_shared<string>("b-tgc"));
		ulteriusParamChanged(make_shared<string>("b-focus count"));
		ulteriusParamChanged(make_shared<string>("focus depth"));
		ulteriusParamChanged(make_shared<string>("color-gain"));
		ulteriusParamChanged(make_shared<string>("color-ensemble"));
		ulteriusParamChanged(make_shared<string>("color-deviation"));
		ulteriusParamChanged(make_shared<string>("color-numDummyLines"));
		ulteriusParamChanged(make_shared<string>("color-prp"));
		ulteriusParamChanged(make_shared<string>("color-freq"));
		ulteriusParamChanged(make_shared<string>("color-rf decimation"));
		ulteriusParamChanged(make_shared<string>("color-ldensity"));
		ulteriusParamChanged(make_shared<string>("rf-rf decimation"));
		ulteriusParamChanged(make_shared<string>("color-color rect"));

		updateImagingParams();
	}

	//void UltrasonixHandler::getAvailableDepths(map<double,int>& depthMapping)
	//{
	//	depthMapping = m_availableDepths;
	//}
	//
	//void UltrasonixHandler::getCurrentDepthIndex(int& depth) 
	//{	
	//	depth = m_availableDepths[m_pCurrentContextMap->m_rxDepthInM];
	//}
	//
	//void UltrasonixHandler::setDepth(int index)
	//{	
	//	std::map<double,int>::iterator it=m_availableDepths.begin();
	//	while (it->second != index && it != m_availableDepths.end()) {
	//		it++;
	//	}
	//
	//	int targetDepth = static_cast<int>(it->first * 1000);
	//	m_pUlterius->setParamValue("b-depth", targetDepth);
	//	Q_EMIT(newVectorMetaData(qSharedPointerCast<RecordObject>(m_pCurrentContextMap)));
	//}

	//void UltrasonixHandler::getImagingMode(USImagingMode& mode) 
	//{
	//	m_accessMutex.lock();
	//	mode = m_imagingMode;
	//	m_accessMutex.unlock();
	//}
	//
	//void UltrasonixHandler::getRFStreamState(bool& rfStreamEnabled) 
	//{
	//	m_accessMutex.lock();
	//	rfStreamEnabled = m_bRFStreamEnabled;
	//	m_accessMutex.unlock();
	//}



	//

	//void UltrasoundInterfaceUltrasonix::setRFStreaming(bool rfEnabled)
	//{
	/*std::lock_guard<std::mutex> lock(m_objectMutex);
	if (rfEnabled)
	{
	m_imagingMode = usModeBModeRF;
	}
	else
	{
	m_imagingMode = usModeBMode;
	}
	updateDataStreaming(m_imagingMode);*/
	//}

}