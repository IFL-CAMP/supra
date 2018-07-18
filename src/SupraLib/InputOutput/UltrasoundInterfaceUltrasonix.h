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

#ifndef __ULTRASOUNDINTERFACEULTRASONIX_H__
#define __ULTRASOUNDINTERFACEULTRASONIX_H__

#ifdef HAVE_DEVICE_ULTRASONIX

#include <atomic>
#include <memory>
#include <mutex>

#include "AbstractInput.h"
#include "USImage.h"

#include "ulterius.h"

struct UlteriusPacket
{
    void* pData;
    int iType;
    int iSize;
    int iFramenum;
    bool bCine;
    double dTimestamp;
};

namespace supra
{
	//TODO add handling of software-controlled parameter changes
	//TODO add other imaging modes
	class UltrasoundInterfaceUltrasonix : public AbstractInput
	{
	public:
		UltrasoundInterfaceUltrasonix(tbb::flow::graph& graph, const std::string& nodeID);
		~UltrasoundInterfaceUltrasonix();

	public:
		virtual void initializeDevice();
		virtual bool ready();

		virtual std::vector<size_t> getImageOutputPorts() { return{ 0 }; };
		virtual std::vector<size_t> getTrackingOutputPorts() { return{}; };

		virtual void freeze();
		virtual void unfreeze();

	protected:
		virtual void startAcquisition();
		//Needs to be thread safe
		virtual void stopAcquisition();
		//Needs to be thread safe
		virtual void configurationEntryChanged(const std::string& configKey);
		//Needs to be thread safe
		virtual void configurationChanged();

		void readConfiguration();

		//virtual void getImagingMode(USImagingMode& mode);
		//virtual void getRFStreamState(bool& rfStreamEnabled);
		//virtual void getAvailableDepths(map<double,int>& depthMapping);
		//virtual void getCurrentDepthIndex(int& depth);
		//virtual void getAvailableFocalDepths(std::map<double,int>& focalDepthMaping);
		//virtual void getCurrentFocalDepthIndex(int& focalDepth);

		//virtual void getGainOverall2D(int& gainValue); // -1...1
		//virtual void getPowerOverall2D(int& powerValue); // -1...0
		//virtual void getGainOverallCFM(int& gainValue); // -1...1
		//virtual void getPowerOverallCFM(int& powerValue); // -1...0
		//virtual void getDopplerPRF(int& prf); // 500....16000

		/*inline virtual void getType(std::string& deviceID) { deviceID = DEVICE_ULTRASONIX_ULTERIUS; };
		inline virtual std::string getType() { return DEVICE_ULTRASONIX_ULTERIUS; };
	*/

		virtual void toggleFreeze();		
		/*virtual void setImagingMode(USImagingMode mode);
		virtual void setDepth(int index);
		virtual void setFocalDepth(int index);
	*/
	//virtual void setGainOverall2D(int gainValue); // -100...100
	//virtual void setPowerOverall2D(int powerValue); // -100...0
	//virtual void setGainOverallCFM(int gainValue); // -100...100
	//virtual void setPowerOverallCFM(int powerValue); // -100...0
	//virtual void setDopplerPRF(int prf); // 500....16000

	//virtual void setGainSectional(int index, int gainValue); // -100...100

	//virtual void setDopplerWindow(double posPercentX, double posPercentY, double posWidthPercent, double posHeightPercent);
	//virtual void setRFStreaming(bool enabled);

		static bool callbackHandler(void * data, int type, int sz, bool cine, int frmnum);
		static bool callbackParamHandler(void* paramID, int ptX, int ptY);

		void initializeParametersFromDevice();

		bool processUlteriusPacket(std::shared_ptr<UlteriusPacket> packet);
		void ulteriusParamChanged(std::shared_ptr<std::string> paramName);

	protected:
		enum USImagingMode
		{
			usModeBMode = 0,
			usModeCFM = 1,
			usModeBModeRF = 2,
			usModeForce32bit = 0xFFFFFFFF    ///< Force enum to 32-bit.
		};

		void setImagingMode(USImagingMode mode);
		void setRFStreaming(bool rfEnabled);

		//void ulteriusCallback(std::shared_ptr<UlteriusPacket> packet);
		bool ulteriusParamCallback(void* paramID, int ptX, int ptY);

		void manageParameterUpdates();
		void updateImagingParams();
		void setImagingModeInternal();
		void updateDataStreaming(USImagingMode mode);
		void getLastError(const char* file = nullptr, int line = 0);

	private:
		double m_tickTimestamp; // last timestamp where data arrived
		double m_hostToUSOffsetInSec;
		double m_speedOfSound;

		std::atomic_bool m_frozen;

		USImagingMode m_imagingMode;

		size_t m_bmodeNumVectors;
		size_t m_bmodeNumSamples;

		size_t m_rfNumVectors;
		size_t m_rfNumSamples;
		double m_rfSamplingFrequInHz;

		std::shared_ptr<const USImageProperties> m_pImageProperties;
		std::atomic_bool m_initialized;

		bool m_RFStreamEnabled;

		std::mutex m_objectMutex;
		std::mutex m_parameterMutex;
		std::mutex m_queueMutex;

		std::condition_variable m_cvParameterUpdates;
		std::queue<std::shared_ptr<std::string> > m_pParameterQueue;

		ulterius* m_pUlterius;
	};

}

#endif //!HAVE_DEVICE_ULTRASONIX

#endif //!__ULTRASOUNDINTERFACEULTRASONIX_H__