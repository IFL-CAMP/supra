// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
// 	and
//      Christoph Hennerpserger
// 		Email c.hennersperger@tum.de
//
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================


#ifndef __USINTCEPHASONICSCC_H__
#define __USINTCEPHASONICSCC_H__

#ifdef HAVE_DEVICE_CEPHASONICS
#ifdef HAVE_CUDA

#include <atomic>
#include <memory>
#include <mutex>
#include <array>

#include <AbstractInput.h>
#include <USImage.h>
#include <fstream>
#include <cstypes.h>
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
	class BeamEnsembleDef;
}

namespace supra
{
	using namespace ::cs;

	class UsIntCephasonicsCcProc;
	class Beamformer;
	class Sequencer;
	class ScanlineTxParameters3D;
	class USTransducer;

	struct BeamEnsembleTxParameters
	{
		enum PulseType {
			Unipolar,
			Bipolar
		};

		double txVoltage;				// voltage applied for pulse
		PulseType txPulseType;			// configuration of pulse (bipolar or unipolar)
		bool txPulseInversion;			// inverted pulse starts on negative pulse direction

		double txDutyCycle;				// duty cycle (percent) used for pulse
		double txFrequency;				// pulse frequency in MHz
		double txPrf;					// pulse repetition frequency of image in Hz
		size_t txRepeatFiring; 			// number of firings for specified pulse before moving to rx

		

		//Cephasonics allows to automatically repeat the tX pulse from a generated wave table, both options can be chosen/combined
		size_t txNumCyclesCephasonics;	// Automatic tX pulse repetition using Cephasonics API
		size_t txNumCyclesManual;		// Manual pulse repetition during signal construction
	};


	class UsIntCephasonicsCc : public AbstractInput
	{
	public:
		UsIntCephasonicsCc(tbb::flow::graph& graph, const std::string& nodeID);
		virtual ~UsIntCephasonicsCc();

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
		// needs to be thread safe
		virtual void configurationDictionaryChanged(const ConfigurationDictionary& newConfig);



	private:

		cs::PlatformHandle* setupPlatform();
		void setupCsProbe();
		void checkOptions();
		void setupScan();
		void setupRxCopying();
		//void initBeams();
		void createSequence();
		std::pair<size_t, const cs::FrameDef*> createFrame(const std::vector<ScanlineTxParameters3D>* txBeamParams, const std::shared_ptr<USImageProperties> imageProps, const BeamEnsembleTxParameters& txParamsCs, const bool disableRx);
		const BeamEnsembleDef* createBeamEnsembleFromScanlineTxParameter(const BeamEnsembleTxParameters& txEnsembleParams, const vec2s numScanlines, const ScanlineTxParameters3D& txParameters);
		void createFrame();
		std::vector<uint8> createWeightedWaveform(const BeamEnsembleTxParameters& txParams, size_t numTotalEntries, float weight, uint8_t csTxOversample);
		void updateTransducer();
		void setBeamSequenceValueRange(size_t oldBeamSequenceValueRange);
		std::string getBeamSequenceApp(size_t totalSequences, size_t sequenceId); 	// return string with appendix for each beam sequence configuration value

		void readVgaSettings();
		void applyVgaSettings();
		void applyVoltageSetting(const cs::FrameDef* pFrameDef, double newVoltage, bool isUniPolar, bool noCheck = false);
		void checkVoltageSetting(const cs::FrameDef* pFrameDef, double targetVoltage, bool isUnipolar);
		void updateImageProperties();

		std::mutex m_objectMutex;

		bool m_ready;

		size_t m_numReceivedFrames;
		size_t m_numDroppedFrames;
		size_t m_lastFrameNumber; // last unique frame number
		std::vector<bool> m_sequenceFramesReceived; // map of received frames within the overall scan protocol (per sequence)
		size_t m_sequenceNumFrames; // number of defined frames within scan (total number of beamformers)

		//cephasonics specific
		cs::PlatformHandle* m_cPlatformHandle;
		std::unique_ptr<cs::USEngine> m_cUSEngine;
		std::thread m_runEngineThread;
		static bool m_environSet;

		std::unique_ptr<UsIntCephasonicsCcProc> m_pDataProcessor;
		std::unique_ptr<Sequencer> m_pSequencer;

		std::unique_ptr<USTransducer> m_pTransducer;

		// global, one ScanDef and Probe possible
		const cs::ScanDef*        m_pScan;
		const cs::Probe*          m_pProbe;

		// many Frame/SubFrames possible
		std::map<size_t, size_t> m_pFrameMap; // mapping of cusdk subframeIDS (key) to linearized frameIDs (value)
		std::vector<const cs::FrameDef*> m_pFrameDefs;
		std::vector<const cs::SubFrameDef*> m_pSubframeDefs;
		std::vector<BeamEnsembleTxParameters> m_beamEnsembleTxParameters; // CS specific transmit parameters
		//std::vector<double> m_voltage; // voltage is special treatment due to rail settings in interface

		// general system properties
		uint32_t m_numPlatforms;
		uint32_t m_numMuxedChannels;
		uint32_t m_numChannelsTotal;
		uint32_t m_numChannelsPerPlatform;
		uint16_t m_probeMapping;

		// system-wide imaging settings (i.e. identical for all individual firings or images in a sequence)
		std::string m_probeName;
		std::vector<size_t> m_probeElementsToMuxedChannelIndices;
		double m_startDepth;
		double m_endDepth;
		bool   m_processorMeasureThroughput;
		double m_speedOfSound; // [m/s]

		// Todo Support generic number of imaging sequences in config directory and XML reader
		size_t m_numBeamSequences;

		// TX settings (system-wide)
		uint32_t m_systemTxClock;


		// RX frontend (analog and digital) settings  (system-wide)
		std::array<double, 10> m_vgaGain;
		uint16_t m_decimation;
		bool   m_decimationFilterBypass;
		double m_antiAliasingFilterFrequency;
		bool   m_highPassFilterBypass;
		double m_highPassFilterFrequency;
		double m_lowNoiseAmplifierGain;
		double m_inputImpedance;

		// RX backend settings  (system-wide)
		uint32_t m_systemRxClock;
		std::vector<bool> m_rxPlatformsToCopy;
		std::vector<size_t> m_rxPlatformCopyOffset;
		size_t m_rxNumPlatformsToCopy;

		bool m_writeMockData;
		bool m_mockDataWritten;
		std::string m_mockDataFilename;

		//TODO investigate whether use of those helps in reducing required bandwidth
		//TODO OR THIS MIGHT BE THE SOLUTION TO THE BEAM LIMIT!
		std::vector<bool> m_txChannelMap;
		std::vector<bool> m_rxChannelMap;
		std::vector<bool> m_muxElementMap;
		std::vector<bool> m_muxSwitchMap;

	protected:
		friend UsIntCephasonicsCcProc;

		void layoutChanged(cs::ImageLayout& layout);

		/// This method represent the data input handle from respectice Cephasonics-specific processor.
		/// Receives informations about the platform (for multiplatform setups), unique frame IDs, incrementing frame numbers
		/// number of received channels, samples and beams, as well as the raw data as scrambled 12-bit values
		void putData(uint16_t platformIndex, size_t frameIndex, size_t frameNumber, uint32_t numChannels, size_t numSamples, size_t numBeams, uint8_t* data);
	};
}

#endif //HAVE_CUDA
#endif //!HAVE_DEVICE_CEPHASONICS

#endif //!__USINTCEPHASONICSCC_H__
