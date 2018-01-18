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


#ifndef __USINTCEPHASONICSBMODE_H__
#define __USINTCEPHASONICSBMODE_H__

#ifdef HAVE_DEVICE_CEPHASONICS

#include <atomic>
#include <memory>
#include <mutex>

#include <AbstractInput.h>
#include <USImage.h>

namespace cs
{
	class USPlatformMgr;
	class PlatformHandle;
	class ScanDef;
	class USEngine;
	class ScanData;
	class FrameBuffer;
	class ImageLayout;
}
namespace supra
{
	class UsIntCephasonicsBmodeProc;

	class UsIntCephasonicsBmode : public AbstractInput<RecordObject>
	{
	public:
		UsIntCephasonicsBmode(tbb::flow::graph& graph, const std::string& nodeID);
		virtual ~UsIntCephasonicsBmode();

		//Functions to be overwritten
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

	private:
		void readConfiguration();

		cs::PlatformHandle* setupPlatform();

		std::shared_ptr<USImageProperties> m_pImageProperties;

		std::mutex m_objectMutex;

		std::string m_xmlFileName;
		int m_processingStage;
		bool m_rfStreaming;

		bool m_ready;

		//cephasonics specific
		cs::PlatformHandle* m_cPlatformHandle;
		const cs::ScanDef * m_cScanDefiniton;
		std::unique_ptr<cs::USEngine> m_cUSEngine;
		std::thread m_runEngineThread;
		static bool m_environSet;

		std::unique_ptr<UsIntCephasonicsBmodeProc> m_pDataProcessor;

	protected:
		friend UsIntCephasonicsBmodeProc;

		void layoutChanged(cs::ImageLayout& layout);
		void putData(cs::FrameBuffer * frameBuffer);
		void putData(cs::ScanData * scanData);
	};
}

#endif //!HAVE_DEVICE_CEPHASONICS

#endif //!__USINTCEPHASONICSBMODE_H__
