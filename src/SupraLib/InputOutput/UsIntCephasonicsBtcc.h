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


#ifndef __USINTCEPHASONICSBTCC_H__
#define __USINTCEPHASONICSBTCC_H__

#ifdef HAVE_DEVICE_CEPHASONICS

#include <atomic>
#include <memory>
#include <mutex>

#include <AbstractInput.h>
#include <USImage.h>

#include <cstypes.h>
#include <string>
using std::string;
#include <firCoeffs.h>

namespace cs
{
	class USPlatformMgr;
	class PlatformHandle;
	class ScanDef;
	class USEngine;
	class FrameBuffer;
	class ImageLayout;
	class BeamDef;
	class ScanDef;
	class Probe;
	class FrameDef;
	class SubFrameDef;
}

namespace supra
{
	using namespace ::cs;

	class UsIntCephasonicsBtccProc;

	class UsIntCephasonicsBtcc : public AbstractInput
	{
	public:
		UsIntCephasonicsBtcc(tbb::flow::graph& graph, const std::string& nodeID);
		virtual ~UsIntCephasonicsBtcc();

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
		void setupProbe();
		void setProgramOptions();
		void checkOptions();
		void setupScan();
		//vector<bool> createChannelVector(string value, uint32 elements);
		void initBeams();

		std::shared_ptr<USImageProperties> m_pImageProperties;

		std::mutex m_objectMutex;

		double m_voltage;
		double m_intensityScaling;
		std::string m_layout;

		bool m_ready;

		//cephasonics specific
		//cs::USPlatformMgr* m_cPlatformManager;
		cs::PlatformHandle* m_cPlatformHandle;
		std::unique_ptr<cs::USEngine> m_cUSEngine;
		std::thread m_runEngineThread;
		static bool m_environSet;

		std::unique_ptr<UsIntCephasonicsBtccProc> m_pDataProcessor;

		const cs::ScanDef*        _scan;
		const cs::Probe*          _probe;
		const cs::FrameDef*       _frameDef;
		const cs::SubFrameDef*    _subframe;

		int 			      _numPlatforms;
		std::string           _txChannel;
		std::string           _rxChannel;
		std::string           _muxElement;
		std::string           _muxSwitch;
		double                _prf;
		uint16                _numElem;
		uint16                _simuFireNumElem;
		uint16                _maxDump;
		uint32                _probeMapping;
		bool                  _firBypass;
		bool                  _firBypassPresent;
		bool                  _HPFBypass;
		bool                  _HPFBypassPresent;
		double                _endDepth;
		uint16                _clock;
		bool                  _tpEn;
		uint16                _dec;
		bool                  _decPresent;
		double                _startDepth;
		double                _inpimp;
		double                _aaf;
		double                _hpf;
		uint16                _numCycles;
		double                _txFreq;
		bool                  _fullAperture;
		uint16                _vgaGain;
		bool                  _throughput;

		static const std::map<uint16, std::vector<std::vector<double> > > coeffs_map;

		// All these have to be set to SYS_LICENSED_CH
		uint16 SYS_LICENSED_CH;
		uint16 NUM_ELEMENTS;
		uint16 NUM_CUSTOM_BEAMS;
		uint16 NUM_EFF_ELEMENTS;
		const BeamDef* txBeamArr[512];
		const BeamDef* rxBeamArr[512];
		std::vector<bool>*  txPiezoArr[512];

		static std::map<uint16, std::vector<std::vector<double> > > create_coeffs_map()
		{
			std::map<uint16, std::vector<std::vector<double> > > coeffs_map;
			std::vector<std::vector<double> > coeffs2(32);
			std::vector<std::vector<double> > coeffs4(32);
			std::vector<std::vector<double> > coeffs8(32);
			std::vector<std::vector<double> > coeffs16(32);
			std::vector<std::vector<double> > coeffs32(32);
			for (uint32 x = 0; x < 32; x++)
			{
				coeffs2[x].assign(myCoeffs2[x], myCoeffs2[x] + 2 * 8);
				coeffs4[x].assign(myCoeffs4[x], myCoeffs4[x] + 4 * 8);
				coeffs8[x].assign(myCoeffs8[x], myCoeffs8[x] + 8 * 8);
				coeffs16[x].assign(myCoeffs16[x], myCoeffs16[x] + 16 * 8);
				coeffs32[x].assign(myCoeffs32[x], myCoeffs32[x] + 32 * 8);
			}
			coeffs_map[2] = coeffs2;
			coeffs_map[4] = coeffs4;
			coeffs_map[8] = coeffs8;
			coeffs_map[16] = coeffs16;
			coeffs_map[32] = coeffs32;
			return coeffs_map;
		}

	protected:
		friend UsIntCephasonicsBtccProc;

		void layoutChanged(cs::ImageLayout& layout);
		void putData(uint16_t platformIndex, uint32_t numChannels, size_t numSamples, size_t numBeams, int16_t* data);
	};
}

#endif //!HAVE_DEVICE_CEPHASONICS

#endif //!__USINTCEPHASONICSBTCC_H__
