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
		: AbstractInput(graph, nodeID,1)
		, m_tickTimestamp(0)
		, m_imagingMode(usModeBMode)
		, m_frozen(false)
	{
		g_pUSLib = this;
		m_pUlterius = new ulterius();

		m_initialized = false;

		m_pImageProperties = make_shared<USImageProperties>(
			vec2s{ 128, 1 },
			512,
			USImageProperties::ImageType::BMode,
			USImageProperties::ImageState::RF,
			USImageProperties::TransducerType::Linear,
			60);

		//TODO figure out value ranges
		//Preload Value Range Dictionary with ranges of allowed values
		m_valueRangeDictionary.set<string>("remoteIP", "127.0.0.1", "Ultrasonix IP");
		m_valueRangeDictionary.set<bool>("RF", { false, true }, false, "Stream RF");
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
		if (configKey == "RF")
		{
			bool newRF = m_configurationDictionary.get<bool>(configKey, false);
			if (newRF != m_RFStreamEnabled)
			{
				m_RFStreamEnabled = newRF;
				setImagingModeInternal();
				setImagingMode(m_imagingMode);
			}
		}
	}

	void UltrasoundInterfaceUltrasonix::configurationChanged()
	{
		readConfiguration();
	}



	void UltrasoundInterfaceUltrasonix::readConfiguration()
	{
		std::string remoteIP;
		
		remoteIP = m_configurationDictionary.get<string>("remoteIP");
		m_RFStreamEnabled = m_configurationDictionary.get<bool>("RF");
		m_speedOfSound = m_configurationDictionary.get<double>("speedOfSound");

		// Connect to ulterius
		// TODO -> figure out offset via lan
		m_hostToUSOffsetInSec = 0;

		// TODO this should not be started here, but in initializeDevice
		bool initSuccess = true;
		initSuccess &= m_pUlterius->connect((char*)remoteIP.c_str());

		setImagingModeInternal();
		setImagingMode(m_imagingMode);

		//initSuccess &= m_pUlterius->setSharedMemoryStatus(0);
		if (!initSuccess) {
			log_error("Could not connect to Ultrasonix.");
		}


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

	bool UltrasoundInterfaceUltrasonix::processUlteriusPacket(shared_ptr<UlteriusPacket> packet) {

		//std::cout << " frame : " << packet->iFramenum << ", rate:  " << 1.0/(packet->dTimestamp-m_tickTimestamp) << ", ";
		m_tickTimestamp = packet->dTimestamp;


		/*udtScreen                 = 0x00000001,
		udtBPre                   = 0x00000002,
		udtBPost                  = 0x00000004,
		udtBPost32                = 0x00000008,
		udtRF                     = 0x00000010,
		udtMPre                   = 0x00000020,
		udtMPost                  = 0x00000040,
		udtPWRF                   = 0x00000080,
		udtPWSpectrum             = 0x00000100,
		udtColorRF                = 0x00000200,
		udtColorCombined          = 0x00000400,
		udtColorVelocityVariance  = 0x00000800,
		udtElastoCombined         = 0x00002000,
		udtElastoOverlay          = 0x00004000,
		udtElastoPre              = 0x00008000,
		udtECG                    = 0x00010000,
		udtPNG                    = 0x10000000*/

		{
			static CallFrequency m("Ultr");

			lock_guard<mutex> lock(m_objectMutex);
			// evaluate data type
			if (udtBPre == packet->iType && !m_RFStreamEnabled) {
				if (packet->iSize != m_bmodeNumSamples*m_bmodeNumVectors)
				{
					log_log("Ulterius packet did not have expected size. Was: ", packet->iSize, " expected ", m_bmodeNumSamples*m_bmodeNumVectors, " = ", m_bmodeNumVectors, " * ", m_bmodeNumSamples);
					log_warn("Dropping incomplete frame");
					updateImagingParams();
					return false;
				}

				// copy the data into a buffer managed by us (i.e. a shared pointer)
				// and create the image
				auto spData = make_shared<Container<uint8_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), m_bmodeNumSamples*m_bmodeNumVectors);
				memcpyTransposed(spData->get(), (uint8_t*)(packet->pData), m_bmodeNumVectors, m_bmodeNumSamples);
				shared_ptr<USImage<uint8_t> > pImage = make_shared < USImage<uint8_t> >(
					vec2s{ m_bmodeNumVectors, m_bmodeNumSamples }, spData, m_pImageProperties, packet->dTimestamp, packet->dTimestamp + m_hostToUSOffsetInSec);

				m.measure();
				addData<0>(pImage);
			}
			else if (udtRF == packet->iType) {
				if (packet->iSize != m_rfNumSamples*m_rfNumVectors * sizeof(int16_t))
				{
					log_log("Ulterius packet did not have expected size. Was: ", packet->iSize, " expected ", m_rfNumSamples*m_rfNumVectors * sizeof(int16_t), " = ", m_rfNumVectors, " * ", m_rfNumSamples, " * sizeof(int16_t)");
					std::cout << "Dropping incomplete frame" << std::endl;
					//TODO make sure we deal with that correctly
					updateImagingParams();
					return false;
				}

				// copy the data into a buffer managed by us (i.e. a shared pointer)
				// and create the image
				auto spData = make_shared<Container<int16_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), m_rfNumSamples*m_rfNumVectors);
				memcpyTransposed(spData->get(), (int16_t*)(packet->pData), m_rfNumVectors, m_rfNumSamples);
				shared_ptr<USImage<int16_t> > pImage = make_shared < USImage<int16_t> >(
					vec2s{ m_rfNumVectors, m_rfNumSamples }, spData, m_pImageProperties, packet->dTimestamp, packet->dTimestamp + m_hostToUSOffsetInSec);

				m.measure();
				addData<0>(pImage);
			}

			//}  else if (udtColorRF == packet->iType) {

		 //       if (packet->iSize != m_pCurrentContextMap->m_cfmNumVectors*m_pCurrentContextMap->m_cfmNumSamples*m_pCurrentContextMap->m_cfmEnsembleSize*sizeof(int16_t))
		 //       {
		 //           std::cout << "Dropping incomplete frame" << std::endl;
			//		//TODO make sure we deal with that correctly
			//		//updateImagingParams();
		 //           return false;
		 //       }


			//    unsigned int noVectors = m_pCurrentContextMap->m_cfmNumVectors;
		 //       unsigned int noSamples = m_pCurrentContextMap->m_cfmNumSamples;
		 //       unsigned int ensembleSize = m_pCurrentContextMap->m_cfmEnsembleSize;

			//	std::vector<QSharedPointer<USVectorRayCFMRF>> vectorRayData;
			//	for (size_t k=0; k<noVectors; ++k) {
			//		std::shared_ptr<std::vector<std::vector<int16_t>>> raySamples(new std::vector<std::vector<int16_t>>(ensembleSize, std::vector<int16_t>(noSamples, 0)));
			//		std::shared_ptr<std::vector<double>> rayTimestamps(new std::vector<double>(std::vector<double>(ensembleSize, 0)));

			//		for (unsigned int e=0; e<ensembleSize; ++e) {
			//			rayTimestamps->at(e) = packet->dTimestamp;
			//			memcpy(&(raySamples->at(e).front()), (void*)((char*)packet->pData + k * e * noSamples * sizeof(int16_t)), noSamples * sizeof(int16_t));
			//		}
			//		
			//		QSharedPointer<USVectorRayCFMRF> rayData( new USVectorRayCFMRF(raySamples, 
			//			m_numReceivedFrames,
			//			k, 
			//			noSamples, 
			//			ensembleSize, 
			//			rayTimestamps, 
			//			packet->dTimestamp, 
			//			packet->dTimestamp+m_hostToUSOffsetInSec)
			//			);

			//		vectorRayData.push_back(rayData);
			//		Q_EMIT(newVectorRayCFMRF(qSharedPointerCast<RecordObject>(rayData)));
			//	}
			//}
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
		shared_ptr<USImageProperties> imProp = make_shared<USImageProperties>(*m_pImageProperties);

		bool paramOK = true;
		bool needUpdateOfImagingParams = false;
		if (paramName == "mode id")
		{
			//cout << "mode id" << endl;
			// imaging modes
			// 0 = default bmode
			// 12 = bmode with rf possibilites
			// 2 = color doppler
			int imageMode = m_pUlterius->getActiveImagingMode();
			log_log("Ultrasonix: Imaging-Mode: ", imageMode);

			USImagingMode curMode = usModeBMode;
			if (0 == imageMode) {
				curMode = usModeBMode;
			}
			else if (2 == imageMode) {
				curMode = usModeCFM;
			}
			else if (imageMode == 12)
			{
				curMode = usModeBModeRF;
			} else {
				std::cerr << "Error: unsupported mode! please update mode " << imageMode << std::endl;
			}
			if (curMode != m_imagingMode)
			{
				setImagingMode(curMode); // try to get to correct mode
			}
		}
		else if (paramName == "probe id")
		{
			char* tmp = new char[80];
			paramOK &= m_pUlterius->getActiveProbe(tmp, 80);
			if (strncmp(tmp, "C5-2/60", 80) == 0)
			{
				imProp->setTransducerType(USImageProperties::Curved);
				imProp->setScanlineLayout({ 128, 1 });
				//imProp->setProbeRadius(60);
				//imProp->setFov(degToRad(60.0));
				needUpdateOfImagingParams = true;
			}
			else if (strncmp(tmp, "L14-5/38", 80) == 0)
			{
				imProp->setTransducerType(USImageProperties::Linear);
				imProp->setScanlineLayout({ 128, 1 });
				//imProp->setProbeRadius(0);
				//imProp->setFov(0.3*(128 - 1));
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
			imProp->setDepth(static_cast<double>(value));
			log_log("changed b-depth: ", imProp->getDepth());
		}
		else if (paramName == "b-ldensity")
		{
			int value;
			//needUpdateOfImagingParams = true;
			paramOK &= m_pUlterius->getParamValue("b-ldensity", value);
			imProp->setScanlineLayout({ static_cast<size_t>(value), 1 });
			m_bmodeNumVectors = value;
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
			imProp->setSpecificParameter("b-tgc", tgcVector);
			imProp->setSpecificParameter("b-tgc relative depth", tgcDepthVector);
			log_log("changed b-tgc");
		}
		else if (paramName == "rf-mode")
		{
			int value;
		    paramOK &= m_pUlterius->getParamValue("rf-mode", value);
			log_log("Ultrasonix: rf-mode changed: ", value);
		}
		//  else if (paramName == "color-gain") 
		//  {
		//  	paramOK &= m_pUlterius->getParamValue("color-gain", value);
		//      m_pCurrentContextMap->m_cfmGainInDB = value;
		//      std::cout << "color-gain changed: " << value << std::endl;
		//  }
		//  else if (paramName == "color-ensemble") 
		//  {
		//      paramOK &= m_pUlterius->getParamValue("color-ensemble", value);
		//      m_pCurrentContextMap->m_cfmEnsembleSize			= value;
		//      std::cout << "color-ensemble changed: " << value << std::endl;
		//  }
		//  else if (paramName == "color-numDummyLines") 
		//  {
		//      paramOK &= m_pUlterius->getParamValue("color-numDummyLines", value);
		//      m_pCurrentContextMap->m_cfmNumFirings			= m_pCurrentContextMap->m_cfmEnsembleSize + value;
		//      std::cout << "color-numDummyLines changed: " << value << std::endl;
		//  }
		//  else if (paramName == "color-prp") 
		//  {
		//      paramOK &= m_pUlterius->getParamValue("color-prp", value);
		   //   m_pCurrentContextMap->m_cfmPRFInHz				= 1000.0/(static_cast<double>(value)/1000);
		//      std::cout << "color-prp changed: " << value << std::endl;
		//  }
		//  else if (paramName == "color-freq") 
		//  {
		//      paramOK &= m_pUlterius->getParamValue("color-freq", value);
		   //   m_pCurrentContextMap->m_cfmRxFrequInHz			= static_cast<double>(value);
		//      std::cout << "color-freq changed: " << value << std::endl;
		//  } 
		//  else if (paramName == "color-deviation")
		//  {
		//      paramOK &= m_pUlterius->getParamValue("color-deviation", value);
		//      m_pCurrentContextMap->m_cfmSteeringAngleInRad = M_PI*180.0 / (static_cast<double>(value)/1000.0);	
			  //std::cout << "changed color-deviation: " << m_pCurrentContextMap->m_cfmSteeringAngleInRad << std::endl;
		//  }
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
			imProp->setSpecificParameter(paramName, value);
			log_log("changed ", paramName, ": ", value);
		}
		else if (
			paramName == "b-image rect" ||
			paramName == "b-hd-image rect")
		{
			uRect rect;
			paramOK &= m_pUlterius->getParamValue(paramName.c_str(), rect);
			string valueStr("[" + to_string(rect.left) + ", " + to_string(rect.top) + ", " + to_string(rect.right) + ", " + to_string(rect.bottom) + "]");
			imProp->setSpecificParameter<string>(paramName, valueStr);
			log_log("changed ", paramName, ": ", valueStr);
		}
		else if (
			paramName == "origin")
		{
			uPoint point;
			paramOK &= m_pUlterius->getParamValue(paramName.c_str(), point);
			imProp->setSpecificParameter<string>(paramName, "[" + to_string(point.x) + ", " + to_string(point.y) + "]");
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

		m_pImageProperties = imProp;

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


	void UltrasoundInterfaceUltrasonix::setImagingMode(UltrasoundInterfaceUltrasonix::USImagingMode mode)
	{
	
	    // check if the given mode matches the currently set data mode
		int curDataMask =  m_pUlterius->getDataToAcquire();
	    USImagingMode curMode = usModeBMode;
		if (curDataMask & udtColorRF  || curDataMask & udtColorCombined) {
			curMode = usModeCFM;
		}
		else if (curDataMask & udtBPre)
		{
			curMode = usModeBMode;
		}
		else if (curDataMask & udtRF) 
		{
			curMode = usModeBModeRF;
		}
	
	    int newUlteriusImageMode = 0;
		if (usModeBMode == mode) {
			// switch to BMode
			newUlteriusImageMode = 0;
		}
		else if (m_RFStreamEnabled)
		{
			newUlteriusImageMode = 12;
		}
		else if (usModeCFM == mode)
		{
			// switch to Color Doppler
	        newUlteriusImageMode = 2;
		}
	
	    int curUlteriusImageMode = m_pUlterius->getActiveImagingMode();
	    if (curMode != mode || newUlteriusImageMode != curUlteriusImageMode) 
		{ 
			if (!m_pUlterius->selectMode(newUlteriusImageMode))
			{
				log_error("Ultrasonix: Failed to set imaging mode ", newUlteriusImageMode);
				MACROgetLastError;
			}
			m_imagingMode = mode;
	    }
	
	    updateDataStreaming(m_imagingMode);
	}


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
	//


	void UltrasoundInterfaceUltrasonix::setRFStreaming(bool rfEnabled)
	{
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
	}

	void UltrasoundInterfaceUltrasonix::updateDataStreaming(USImagingMode mode) {
	    int dataMask = 0;
		if (usModeBMode == mode) {
			// switch to BMode
			dataMask = udtBPre;
			m_pUlterius->setParamValue("rf-mode", 0);
		}
		else if(mode == usModeBModeRF)
		{
			dataMask = udtRF | udtBPre;
			m_pUlterius->setParamValue("rf-mode", 2);
		} 
		else if (usModeCFM == mode) {
		    m_pUlterius->setParamValue("rf-mode", 2);
	        m_pUlterius->setParamValue("color-rf decimation", 2);
			
	        /*if (rfEnabled)
	        {
	            std::cout << "Warning: simultaneous RF and ColorRF will result in massive dropped frames for CRF" << std::endl;
	        }*/
	
			// switch to Color Doppler
			//dataMask = rfEnabled ? (udtColorRF | udtRF) : (udtColorRF | udtBPre); // works only from 6.07 onwards!
	        //dataMask = udtColorRF;
		}
	
	    // update
	    m_pUlterius->setDataToAcquire(dataMask);
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


	void UltrasoundInterfaceUltrasonix::updateImagingParams()
	{
		int dataMask;
		if (m_imagingMode == usModeBMode) 
		{
			dataMask = udtBPre;

			uDataDesc dataDescriptor;
			if (m_pUlterius->getDataDescriptor((uData)dataMask, dataDescriptor))
			{
				m_bmodeNumVectors = dataDescriptor.w;
				m_bmodeNumSamples = dataDescriptor.h;

				shared_ptr<USImageProperties> imProp = make_shared<USImageProperties>(*m_pImageProperties);
				imProp->setNumSamples(m_bmodeNumSamples);
				imProp->setScanlineLayout({ m_bmodeNumVectors, 1 });
				imProp->setImageState(USImageProperties::ImageState::PreScan);
				m_pImageProperties = imProp;
				//->m_bmodeSamplesPerM = m_pCurrentContextMap->m_bmodeNumSamples / m_pCurrentContextMap->m_rxDepthInM;

				//double samplesPerM = 2*m_pCurrentContextMap->m_bmodeSamplingFrequInHz / m_pCurrentContextMap->m_tissueSpeedInMPerS;
				//m_pCurrentContextMap->m_bmodeSamplesPerM = samplesPerM;

				log_log("Ultrasonix: New image params - bmode vectors: ", m_bmodeNumVectors, ", samples: ", m_bmodeNumSamples);
			}
			else
			{
				log_error("Error fetching data descriptor.");
				MACROgetLastError;
			}
		}
		else if (m_imagingMode == usModeBModeRF)
		{
			dataMask = udtRF;
			uDataDesc dataDescriptor;
			//	std::cout << "Warning: Color RF data not thoroughly tested up to now." << std::endl;
			if (m_pUlterius->getDataDescriptor((uData)dataMask, dataDescriptor))
			{
				m_rfNumVectors = dataDescriptor.w;
				m_rfNumSamples = dataDescriptor.h;

				shared_ptr<USImageProperties> imProp = make_shared<USImageProperties>(*m_pImageProperties);
				imProp->setNumSamples(m_rfNumSamples);
				imProp->setScanlineLayout({ m_rfNumVectors, 1 });
				imProp->setImageState(USImageProperties::ImageState::RF);
				m_pImageProperties = imProp;

				//m_pCurrentContextMap->m_rfSamplesPerM = m_pCurrentContextMap->m_rfNumSamples / m_pCurrentContextMap->m_rxDepthInM;

				double samplesPerM = 2 * m_rfSamplingFrequInHz / m_speedOfSound;
				//m_pCurrentContextMap->m_rfSamplesPerM = samplesPerM;
				log_log("Ultrasonix: New image params - rf vectors: ", m_rfNumVectors, ", samples: ", m_rfNumSamples);
			}
			else
			{
				log_error("Error fetching data descriptor.");
				MACROgetLastError;
			}
		}
		//   if (m_imagingMode == usModeCFM) 
		//   {
		   //	dataMask = udtColorRF;	

		   //	uDataDesc dataDescriptor;
		   //	if (m_pUlterius->getDataDescriptor( (uData)dataMask, dataDescriptor )) 
		//       {
		   //		m_pCurrentContextMap->m_cfmNumVectors = dataDescriptor.w;
		   //		m_pCurrentContextMap->m_cfmNumSamples = dataDescriptor.h;
		   //		//m_pCurrentContextMap->m_cfmSamplesPerM = m_pCurrentContextMap->m_cfmNumSamples / m_pCurrentContextMap->m_rxDepthInM;
		//           double samplesPerM = 2*m_pCurrentContextMap->m_cfmSamplingFrequInHz / m_pCurrentContextMap->m_tissueSpeedInMPerS;
		//           m_pCurrentContextMap->m_cfmSamplesPerM = samplesPerM;

		//           std::cout << "UltrasonixHandler(): New image params - cfm vectors: " << m_pCurrentContextMap->m_cfmNumVectors << ", samples: " << m_pCurrentContextMap->m_cfmNumSamples << std::endl;
		   //	}

		   //	//
		   //}
	}

	void UltrasoundInterfaceUltrasonix::setImagingModeInternal()
	{
		if (m_RFStreamEnabled)
		{
			m_imagingMode = usModeBModeRF;
		}
		else
		{
			m_imagingMode = usModeBMode;
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

}