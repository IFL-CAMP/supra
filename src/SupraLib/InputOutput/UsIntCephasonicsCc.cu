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

#include "UsIntCephasonicsCcProc.h"
#include "UsIntCephasonicsCc.h"

#include <memory>

#include "utilities/cudaUtility.h"

#include <USPlatformMgr.h>
#include <PlatformHandle.h>
#include <ScanDef.h>
#include <USEngine.h>
#include <FrameBuffer.h>
#include <FrameDef.h>
#include <AddImageLayout.h>
#include <AddScanConverter.h>
#include <EnableScanConverter.h>

#include <LinearProbe.h>
#include <BeamEnsembleDef.h>
#include <BeamDef.h>
#include <TXBeamDef.h>
#include <RXBeamDef.h>
#include <FiringDef.h>

#include <cmd/AddFrame.h>
#include <cmd/SetClock.h>
#include <cmd/SetTGC.h>
#include <cmd/SetLNA.h>
#include <cmd/SetAAF.h>
#include <cmd/SetDecimFiltBypass.h>
#include <cmd/SetDecimFiltDecimation.h>
#include <cmd/SetDecimFiltCoeffs.h>
#include <cmd/SetTransmitVoltage.h>
#include <cmd/GetTransmitVoltage.h>
#include <cmd/SetTimeout.h>
#include <cmd/SetPRF.h>
#include <cmd/SetTPG.h>
#include <cmd/SetHPF.h>
#include <cmd/SetTXChannel.h>
#include <cmd/SetRXChannel.h>
#include <cmd/SetMuxElement.h>
#include <cmd/SetMuxSwitch.h>

#include <set>

#include "USImage.h"
#include "Beamformer/USRawData.h"
#include "Beamformer/Sequencer.h"
#include "Beamformer/USTransducer.h"
#include "utilities/utility.h"
#include "utilities/CallFrequency.h"
#include "utilities/Logging.h"
#include "ContainerFactory.h"

namespace supra
{
	using namespace std;
	using namespace ::cs;
	using ::cs::USPlatformMgr;
	using logging::log_error;
	using logging::log_log;

	bool UsIntCephasonicsCc::m_environSet = false;

	union CephasonicsRawData4Channels
	{
		uint8_t raw[6]; // 48 bit
		struct
		{  // This weird order is due to x86 beeing little-endian
			int64_t c0 :12;
			int64_t c1 :12;
			int64_t c2 :12;
			int64_t c3 :12;
		} ordered;
	};

	__global__ void copyUnscramble(
			size_t numBeams,
			size_t numSamples,
			size_t numChannels,
			size_t numBytesChannels,
			size_t platformOffset,
			size_t numPlatformsToCopy,
			const uint8_t* __restrict__ dataScrambled,
			int16_t* __restrict__ dataUnscrambled)
	{
		int beam = blockDim.y * blockIdx.y + threadIdx.y; //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		int sample = blockDim.x * blockIdx.x + threadIdx.x; //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		if(beam < numBeams && sample < numSamples)
		{
			for(int channel = 0; channel < (numChannels/4); channel ++)
			{
				int channelOut;
				if(channel*4 < 16)
				{
					channelOut = channel*4;
				}
				else if(channel*4 < 32)
				{
					channelOut = channel*4 + 16;
				}
				else if(channel*4 < 48)
				{
					channelOut = channel*4 - 16;
				}
				else {
					channelOut = channel*4;
				}
				CephasonicsRawData4Channels d;
				d.raw[0] = dataScrambled[ channel*6 +     sample*numBytesChannels + beam*numBytesChannels*numSamples];
				d.raw[1] = dataScrambled[ channel*6 + 1 + sample*numBytesChannels + beam*numBytesChannels*numSamples];
				d.raw[2] = dataScrambled[ channel*6 + 2 + sample*numBytesChannels + beam*numBytesChannels*numSamples];
				d.raw[3] = dataScrambled[ channel*6 + 3 + sample*numBytesChannels + beam*numBytesChannels*numSamples];
				d.raw[4] = dataScrambled[ channel*6 + 4 + sample*numBytesChannels + beam*numBytesChannels*numSamples];
				d.raw[5] = dataScrambled[ channel*6 + 5 + sample*numBytesChannels + beam*numBytesChannels*numSamples];

				int16_t o1 = d.ordered.c0;
				int16_t o2 = d.ordered.c1;
				int16_t o3 = d.ordered.c2;
				int16_t o4 = d.ordered.c3;

				dataUnscrambled[sample + (channelOut + 0 + platformOffset*numChannels)*numSamples +
								beam*numChannels*numPlatformsToCopy*numSamples] = o1;
				dataUnscrambled[sample + (channelOut + 1 + platformOffset*numChannels)*numSamples +
								beam*numChannels*numPlatformsToCopy*numSamples] = o2;
				dataUnscrambled[sample + (channelOut + 2 + platformOffset*numChannels)*numSamples +
								beam*numChannels*numPlatformsToCopy*numSamples] = o3;
				dataUnscrambled[sample + (channelOut + 3 + platformOffset*numChannels)*numSamples +
								beam*numChannels*numPlatformsToCopy*numSamples] = o4;
			}
		}
	}


	UsIntCephasonicsCc::UsIntCephasonicsCc(tbb::flow::graph & graph, const std::string& nodeID)
		: AbstractInput(graph, nodeID, 2)
		, m_pTransducer(nullptr)
		, m_pSequencer(nullptr)
		, m_pProbe(nullptr)
		, m_cUSEngine(nullptr)
		, m_numMuxedChannels(0)
		, m_numChannelsTotal(0)
		, m_probeMapping(0)
		, m_systemRxClock(40)
		, m_mockDataWritten(false)
		, m_numBeamSequences(0) // TODO replace hardcoded sequence number (move to config/gui)
		, m_numReceivedFrames(0)
		, m_numDroppedFrames(0)
		, m_lastFrameNumber(1)
		, m_sequenceNumFrames(0)
	{
		m_callFrequency.setName("CepUS");

		m_ready = false;

		if(!m_environSet)
		{
			setenv("CS_CCMODE", "1", true);
			setenv("CS_TX_PUSH", "1", true);
			setenv("CS_LOGFILE", "", true);

			m_environSet = true;
		}

		m_cPlatformHandle = setupPlatform();

		// number of different frames (e.g. 1 PW follwed by 1 B-Mode) as basic determinator for other settings
		m_valueRangeDictionary.set<uint32_t>("sequenceNumFrames", 1, 10, 1, "Number of different beam sequences");
		m_numBeamSequences = 1; // not known at startup as this will be updated only once xml configuration is read

		//Setup allowed values for parameters
		m_valueRangeDictionary.set<uint32_t>("systemTxClock", {40, 20}, 40, "TX system clock (MHz)");
		m_valueRangeDictionary.set<string>("probeName", {"Linear", "2D", "CPLA12875", "CPLA06475"}, "Linear", "Probe");
		
		m_valueRangeDictionary.set<double>("startDepth", 0.0, 300.0, 0.0, "Start depth [mm]");
		m_valueRangeDictionary.set<double>("endDepth", 0.0, 300.0, 70.0, "End depth [mm]");
		m_valueRangeDictionary.set<bool>("measureThroughput", {false, true}, false, "Measure throughput");
		m_valueRangeDictionary.set<double>("speedOfSound", 1000, 3000, SP_SOUND, "Speed of sound [m/s]");



		m_valueRangeDictionary.set<double>("tgc0", 0.0, 100.0, 25.0, "TGC 0 [dB]");
		m_valueRangeDictionary.set<double>("tgc1", 0.0, 100.0, 40.0, "TGC 1 [dB]");
		m_valueRangeDictionary.set<double>("tgc2", 0.0, 100.0, 35.0, "TGC 2 [dB]");
		m_valueRangeDictionary.set<double>("tgc3", 0.0, 100.0, 40.0, "TGC 3 [dB]");
		m_valueRangeDictionary.set<double>("tgc4", 0.0, 100.0, 45.0, "TGC 4 [dB]");
		m_valueRangeDictionary.set<double>("tgc5", 0.0, 100.0, 47.0, "TGC 5 [dB]");
		m_valueRangeDictionary.set<double>("tgc6", 0.0, 100.0, 47.0, "TGC 6 [dB]");
		m_valueRangeDictionary.set<double>("tgc7", 0.0, 100.0, 50.0, "TGC 7 [dB]");
		m_valueRangeDictionary.set<double>("tgc8", 0.0, 100.0, 50.0, "TGC 8 [dB]");
		m_valueRangeDictionary.set<double>("tgc9", 0.0, 100.0, 50.0, "TGC 9 [dB]");
		m_valueRangeDictionary.set<uint32_t>("decimation", {1, 2, 4, 8, 16, 32}, 1, "Input decimation");
		m_valueRangeDictionary.set<bool>("decimationFilterBypass", {false, true}, true, "DecimationFilter bypass");
		m_valueRangeDictionary.set<double>("antiAliasingFilterFrequency", 3, 18, 15, "AntiAliasFilter [MHz]");
		m_valueRangeDictionary.set<bool>("highPassFilterBypass", {false, true}, true, "HighPassFilter bypass");
		m_valueRangeDictionary.set<double>("highPassFilterFrequency", 0.14, 1.4, 0.898, "HighPassFilter [MHz]");
		m_valueRangeDictionary.set<double>("lowNoiseAmplifierGain", 10, 20, 18.5, "LowNoiseAmp gain [dB]");
		m_valueRangeDictionary.set<double>("inputImpedance", {50,200}, 200, "Input impedance [Ohm]");

		m_valueRangeDictionary.set<bool>("writeMockData", {false, true}, false, "(Write mock)");
		m_valueRangeDictionary.set<string>("mockDataFilename", "", "(Mock meta filename)");

		// set beamSeq specific value ranges, too.
		setBeamSequenceValueRange(0);


		// create new sequencer
		m_pSequencer = unique_ptr<Sequencer>(new Sequencer(m_numBeamSequences));
		m_beamEnsembleTxParameters.resize(m_numBeamSequences);

		configurationChanged();
	}

	UsIntCephasonicsCc::~UsIntCephasonicsCc()
	{
		if (m_cUSEngine)
		{
			//End of the world, waiting to join completed thread on teardown
			m_cUSEngine->tearDown();   //Teardown USEngine
		}
		if (m_runEngineThread.joinable())
		{
			m_runEngineThread.join();
		}
	}

	// return appendix for configuration strings for a given beam sequence
	// consider backward compatibility for a single beam sequence, where the appendix strings are simply ommited
	std::string UsIntCephasonicsCc::getBeamSequenceApp(size_t totalSequences, size_t sequenceId)
	{
		if (totalSequences == 1)
		{
			return std::string("");
		}
		else
		{
			return "seq"+std::to_string(sequenceId)+"_";
		}
	}

	void UsIntCephasonicsCc::setBeamSequenceValueRange(size_t oldBeamSequenceValueRange)
	{
		for (size_t numSeq = 0; numSeq < oldBeamSequenceValueRange; ++numSeq)
		{
				// remove old keys
				std::string idApp = getBeamSequenceApp(oldBeamSequenceValueRange,numSeq);

				m_valueRangeDictionary.remove(idApp+"scanType");
				m_valueRangeDictionary.remove(idApp+"rxModeActive");
				m_valueRangeDictionary.remove(idApp+"txVoltage");
				m_valueRangeDictionary.remove(idApp+"txPulseType");
				m_valueRangeDictionary.remove(idApp+"txPulseInversion");
				m_valueRangeDictionary.remove(idApp+"txFrequency");
				m_valueRangeDictionary.remove(idApp+"txPulseRepetitionFrequency");
				m_valueRangeDictionary.remove(idApp+"txPulseRepeatFiring");
				m_valueRangeDictionary.remove(idApp+"txWindowType");
				m_valueRangeDictionary.remove(idApp+"txWindowParameter");
				m_valueRangeDictionary.remove(idApp+"txDutyCycle");
				m_valueRangeDictionary.remove(idApp+"txNumCyclesCephasonics");
				m_valueRangeDictionary.remove(idApp+"txNumCyclesManual");
				m_valueRangeDictionary.remove(idApp+"numScanlinesX");
				m_valueRangeDictionary.remove(idApp+"numScanlinesY");
				m_valueRangeDictionary.remove(idApp+"rxScanlineSubdivisionX");
				m_valueRangeDictionary.remove(idApp+"rxScanlineSubdivisionY");
				m_valueRangeDictionary.remove(idApp+"txSectorAngleX");
				m_valueRangeDictionary.remove(idApp+"txSectorAngleY");
				m_valueRangeDictionary.remove(idApp+"txSteeringAngleX");
				m_valueRangeDictionary.remove(idApp+"txSteeringAngleY");
				m_valueRangeDictionary.remove(idApp+"apertureSizeX");
				m_valueRangeDictionary.remove(idApp+"apertureSizeY");
				m_valueRangeDictionary.remove(idApp+"txApertureSizeX");
				m_valueRangeDictionary.remove(idApp+"txApertureSizeY");
				m_valueRangeDictionary.remove(idApp+"txFocusActive");
				m_valueRangeDictionary.remove(idApp+"txFocusDepth");
				m_valueRangeDictionary.remove(idApp+"txFocusWidth");
				m_valueRangeDictionary.remove(idApp+"txCorrectMatchingLayers");
				m_valueRangeDictionary.remove(idApp+"numSamplesRecon");
		}


		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			std::string idApp = getBeamSequenceApp(m_numBeamSequences,numSeq);
			
			// make nice string for GUI descriptors
			std::string descApp;
			if (m_numBeamSequences == 1)
			{
				descApp == "";
			}
			else
			{
				descApp = "Seq "+std::to_string(numSeq)+": ";
			}

			// overall scan type for sequence
			m_valueRangeDictionary.set<string>(idApp+"scanType", {"linear", "phased", "biphased", "planewave"}, "linear", descApp+"Scan Type");
			m_valueRangeDictionary.set<bool>(idApp+"rxModeActive", {false, true}, true, descApp+"Activate Rx mode");

			// beam specific settings
			m_valueRangeDictionary.set<double>(idApp+"txVoltage", 6, 140, 6, descApp+"Pulse voltage [V]");
			m_valueRangeDictionary.set<string>(idApp+"txPulseType", {"unipolar", "bipolar"}, "bipolar", descApp+"Pulse Type");
			m_valueRangeDictionary.set<bool>(idApp+"txPulseInversion", {false, true}, false, descApp+"Pulse Inversion [negative V]");
			m_valueRangeDictionary.set<double>(idApp+"txFrequency", 0.0, 20.0, 7.0, descApp+"Pulse frequency [MHz]");
			m_valueRangeDictionary.set<double>(idApp+"txPulseRepetitionFrequency", 0.0, 10000.0, 0.0, descApp+"Pulse repetition frequency [Hz]");
			m_valueRangeDictionary.set<uint32_t>(idApp+"txPulseRepeatFiring", 1, 255, 1, descApp+"Number of Firings");
			m_valueRangeDictionary.set<double>(idApp+"txDutyCycle", 0.0, 1.0, 1.0, descApp+"Duty cycle [percent]");
			m_valueRangeDictionary.set<uint32_t>(idApp+"txNumCyclesCephasonics", 1, 20000, 1, descApp+"Number Pulse Cycles (ceph)");
			m_valueRangeDictionary.set<uint32_t>(idApp+"txNumCyclesManual", 1, 10, 1, descApp+"Number Pulse Cycles (manual)");

			// beam ensemble specific settings
			m_valueRangeDictionary.set<uint32_t>(idApp+"numScanlinesX", 1, 512, 256, descApp+"Number of scanlines X");
			m_valueRangeDictionary.set<uint32_t>(idApp+"numScanlinesY", 1, 512, 1, descApp+"Number of scanlines Y");
			m_valueRangeDictionary.set<uint32_t>(idApp+"rxScanlineSubdivisionX", 1, 512, 256, descApp+"Rx scanline supersampling X");
			m_valueRangeDictionary.set<uint32_t>(idApp+"rxScanlineSubdivisionY", 1, 512, 1, descApp+"Rx scanline supersampling Y");
			m_valueRangeDictionary.set<double>(idApp+"txSectorAngleX", -178, 178, 60, descApp+"Opening angle X [deg]");
			m_valueRangeDictionary.set<double>(idApp+"txSectorAngleY", -178, 178, 60, descApp+"Opening angle Y [deg]");
			m_valueRangeDictionary.set<double>(idApp+"txSteeringAngleX", -90.0, 90.0, 0.0, descApp+"Angle of image X [deg]");
			m_valueRangeDictionary.set<double>(idApp+"txSteeringAngleY", -90.0, 90.0, 0.0, descApp+"Angle of image Y [deg]");
			m_valueRangeDictionary.set<uint32_t>(idApp+"apertureSizeX", 0, 384, 0, descApp+"Aperture X");
			m_valueRangeDictionary.set<uint32_t>(idApp+"apertureSizeY", 0, 384, 0, descApp+"Aperture Y");
			m_valueRangeDictionary.set<uint32_t>(idApp+"txApertureSizeX", 0, 384, 0, descApp+"TX Aperture X");
			m_valueRangeDictionary.set<uint32_t>(idApp+"txApertureSizeY", 0, 384, 0, descApp+"TX Aperture Y");
			m_valueRangeDictionary.set<string>(idApp+"txWindowType", {"Rectangular", "Hann", "Hamming","Gauss"}, "Rectangular", descApp+"TX apodization");
			m_valueRangeDictionary.set<double>(idApp+"txWindowParameter", 0.0, 10.0, 0.0, descApp+"TxWindow parameter");			
			m_valueRangeDictionary.set<bool>(idApp+"txFocusActive", {false, true}, true, descApp+"TX focus");
			m_valueRangeDictionary.set<double>(idApp+"txFocusDepth", 0.0, 300.0, 50.0, descApp+"Focus depth [mm]");
			m_valueRangeDictionary.set<double>(idApp+"txFocusWidth", 0.0, 20.0, 0.0, descApp+"Focus width [mm]");
			m_valueRangeDictionary.set<bool>(idApp+"txCorrectMatchingLayers", {true, false}, false, descApp+"TX matching layer correction");
			m_valueRangeDictionary.set<uint32_t>(idApp+"numSamplesRecon", 10, 4096, 1000, descApp+"Number of samples Recon");

		}
	}

	//TODO replace CS_THROW with something more intelligent

	//Step 1 ------------------ "Setup Platform"
	PlatformHandle* UsIntCephasonicsCc::setupPlatform()
	{
		PlatformHandle* discoveredPlatforms[USPlatformMgr::MAX_HANDLES]; 
		int numPlatforms;

		try 
		{
			numPlatforms = USPlatformMgr::instance()->discoverPlatforms(discoveredPlatforms);
		}
		catch(const cs::csException& e)
		{
			log_error("UsIntCephasonicsCc: Exception during USPlatformMgr::discoverPlatforms. Message: ",
					e.what());
			// how to handle? for now we simply abort.
		}
		if (numPlatforms < 1 || numPlatforms > (int)USPlatformMgr::MAX_HANDLES)
		{
			log_error("UsIntCephasonicsCc: Number of platforms is out of range, numPlatforms=", numPlatforms);
		}
		if (numPlatforms == 0)
		{
			log_error("UsIntCephasonicsCc: No Platforms connected, numPlatforms=", numPlatforms);
		}
		PlatformHandle* ph = discoveredPlatforms[0];

		PlatformCapabilities pC = USPlatformMgr::getPlatformCapabilities(*ph);

		m_numPlatforms = (1+USPlatformMgr::instance()->getNumSlaves(*ph));
		m_numMuxedChannels = pC.SYS_LICENSED_CHANNELS * pC.MUX_RATIO;
		m_numChannelsTotal = pC.SYS_LICENSED_CHANNELS;
		m_numChannelsPerPlatform = m_numChannelsTotal / m_numPlatforms;
		log_log("UsIntCephasonicsCc: Number of connected platforms: ", m_numPlatforms);
		log_log("Licensed channels: ", pC.SYS_LICENSED_CHANNELS, ", Mux ratio: ", pC.MUX_RATIO, ", MAX_RX_BEAMS_SWMODE: ", pC.MAX_RX_BEAMS_SWMODE);
	
		return ph;
	}

	void UsIntCephasonicsCc::setupCsProbe()
	{
		//TODO is the probe object necessary in CC mode?
		//TODO setup Probe depending on selected probe
		string probeName = "BlinkCC";        //Probe Name
		string probeSN = "Unknown";          //Probe Serial No
		double minFreq = 2000000;            //Probe MinFrequency
		double maxFreq = 7000000;            //Probe MaxFrequency
		double maxVoltage = 140;             //Probe MaxVoltage
		double yLen = 0;                     //Probe YLen, not used currently
		uint32 xNumElem = m_numMuxedChannels;
											 //maximum elelments system can support
		double xLen = 2.53e-2;               //Probe Width is irrelevent for custom beam scans
		uint32 yNumElem = 1;                 //Probe yNumElem

		m_pProbe = (Probe*)&LinearProbe::createProbe(*m_cPlatformHandle,probeSN,probeName,
												   minFreq,maxFreq,maxVoltage,m_probeMapping,
												   xLen,xNumElem,yLen,yNumElem);
	}

	void UsIntCephasonicsCc::checkOptions()
	{
		if (m_endDepth > 300)
		{
			CS_THROW("EndDepth can not exceed 300");
		}
	}

	void UsIntCephasonicsCc::updateTransducer() {
		//create new transducer
		vec2s maxAperture{0,0};
		if (m_probeName == "Linear" || m_probeName == "CPLA12875") {
			double probePitch = 0.295275591; // From Cephasonics xscan file
			m_pTransducer = unique_ptr<USTransducer>(
					new USTransducer(
							128,
							vec2s{128,1},
							USTransducer::Linear,
							vector<double>(128 - 1, probePitch),
							vector<double>(0),
							vector<std::pair<double, double> >{{8.9e-05 * 1000.0, 3720.0 * 1000.0}, {7.6e-05 * 1000.0, 2660 * 1000.0}}));

			maxAperture = {64,1};

			if (m_numMuxedChannels == 64 && m_numChannelsTotal == 64)
			{
				// we have a single 64 channel system without muxing
				m_probeElementsToMuxedChannelIndices.resize(64);
				
				for(size_t probeElem = 0; probeElem < 64; probeElem++)
				{
					m_probeElementsToMuxedChannelIndices[probeElem] = probeElem;
				}
			}
			else if ((m_numMuxedChannels == 192 || m_numMuxedChannels == 128) && m_numChannelsTotal == 64)
			{
				// 64 channel system which allows for full realtime imaging of 128 element probe
				m_probeElementsToMuxedChannelIndices.resize(128);

				// Linear transducer is expected to be at platform 0
				// and it uses the first 2 mux switches from each element
				for(size_t probeElem = 0; probeElem < 128; probeElem++)
				{
					m_probeElementsToMuxedChannelIndices[probeElem] = probeElem;
				}
			}
			else if (m_numMuxedChannels == 1152 && m_numChannelsTotal == 384)
			{
				// 384 channel system which allows for full realtime imaging of 128 element probe
				m_probeElementsToMuxedChannelIndices.resize(128);

				// Linear transducer is expected to be at platform 2, that is at connector C
				// and it uses the first 2 mux switches from each element
				for(size_t probeElem = 0; probeElem < 64; probeElem++)
				{
					m_probeElementsToMuxedChannelIndices[probeElem] = probeElem + 128;
				}
				for(size_t probeElem = 64; probeElem < 128; probeElem++)
				{
					m_probeElementsToMuxedChannelIndices[probeElem] = probeElem + 64 + 384;
				}
			}
		} 
		else if (m_probeName == "CPLA06475")
		{
			// Linear array with 64 elements
			double probePitch = 0.295275591; // From Cephasonics xscan file
			m_pTransducer = unique_ptr<USTransducer>(
					new USTransducer(
							64,
							vec2s{64,1},
							USTransducer::Linear,
							vector<double>(64 - 1, probePitch),
							vector<double>(0),
							vector<std::pair<double, double> >{{8.9e-05 * 1000.0, 3720.0 * 1000.0}, {7.6e-05 * 1000.0, 2660 * 1000.0}})
						);

			maxAperture = {64,1};

			if (m_numMuxedChannels == 64 && m_numChannelsTotal == 64)
			{
				// we have a single 64 channel system without muxing
				m_probeElementsToMuxedChannelIndices.resize(64);
				
				for(size_t probeElem = 0; probeElem < 64; probeElem++)
				{
					m_probeElementsToMuxedChannelIndices[probeElem] = probeElem;
				}
			}
			else if (m_numMuxedChannels == 192 && m_numChannelsTotal == 64)
			{
				log_error("UsIntCephasonicsCc: 64 element array not yet supported in 64 channel system with muxing.");
			}
			else if (m_numMuxedChannels == 1152 && m_numChannelsTotal == 384)
			{
				log_error("UsIntCephasonicsCc: 64 element array not yet supported in 384 channel system.");
			}
		} 
		else if (m_probeName == "2D") {
			double probePitch = 0.3; // From Vermon specification
			m_pTransducer = unique_ptr<USTransducer>(
					new USTransducer(
							1024,
							vec2s{32,32},
							USTransducer::Planar,
							vector<double>(32 - 1, probePitch),
							{
									probePitch, probePitch, probePitch, probePitch, probePitch, probePitch, probePitch,
									2*probePitch,
									probePitch, probePitch, probePitch, probePitch, probePitch, probePitch, probePitch,
									2*probePitch,
									probePitch, probePitch, probePitch, probePitch, probePitch, probePitch, probePitch,
									2*probePitch,
									probePitch, probePitch, probePitch, probePitch, probePitch, probePitch, probePitch
							}));

			maxAperture = {32,12};
			m_probeElementsToMuxedChannelIndices.resize(1024);

			// The 2D array uses the element in contiguous order. Nothing to see here.
			for(size_t probeElem = 0; probeElem < 1024; probeElem++)
			{
				m_probeElementsToMuxedChannelIndices[probeElem] = probeElem;
			}
		}

		m_pSequencer->setTransducer(m_pTransducer.get());


		// TODO: currently all beamformers share same aperture
		for (auto numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			
			std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq);

			vec2s bfTxApertureSize = bf->getTxApertureSize();
			vec2s bfApertureSize = bf->getApertureSize();

			if(bfTxApertureSize.x == 0)
			{
				bfTxApertureSize.x = maxAperture.x;
			}
			if(bfTxApertureSize.y == 0)
			{
				bfTxApertureSize.y = maxAperture.y;
			}

			if(bfApertureSize.x == 0)
			{
				bfApertureSize.x = maxAperture.x;
			}
			if(bfApertureSize.y == 0)
			{
				bfApertureSize.y = maxAperture.y;
			}
			bfApertureSize = min(bfApertureSize, maxAperture);
			bfTxApertureSize = min(bfTxApertureSize, maxAperture);

			bf->setTxMaxApertureSize(bfTxApertureSize);
			bf->setMaxApertureSize(bfApertureSize);


			// (re)compute the internal TX parameters for a beamformer if any parameter changed
			if (!bf->isReady())
			{
				bf->computeTxParameters();
			}
		}
	}

	void UsIntCephasonicsCc::updateImageProperties() {

		// iterate over all defined beam sequences, each beam-sequ defines one USImageProperties object
		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			

			std::shared_ptr<const Beamformer> bf = m_pSequencer->getBeamformer(numSeq);
			std::shared_ptr<const USImageProperties> imageProps = m_pSequencer->getUSImgProperties(numSeq);

			vec2s numScanlines = bf->getNumScanlines();
			size_t numDepths = bf->getNumDepths();
			vec2s rxScanlines = bf->getNumRxScanlines();
			vec2 steeringAngle = bf->getTxSteeringAngle();
			vec2 sectorAngle = bf->getTxSectorAngle();
			vec2s apertureSize = bf->getApertureSize();
			vec2s txApertureSize = bf->getTxApertureSize();


			auto newProps = make_shared<USImageProperties>(
				numScanlines,
				numDepths,
				USImageProperties::ImageType::BMode,
				USImageProperties::ImageState::Raw,
				USImageProperties::TransducerType::Linear,
				m_endDepth );

			newProps->setImageType(USImageProperties::ImageType::BMode);				// Defines the type of information contained in the image
			newProps->setImageState(USImageProperties::ImageState::Raw);				// Describes the state the image is currently in
			newProps->setScanlineLayout(numScanlines);					// number of scanlines acquired
			newProps->setDepth(m_endDepth);								// depth covered

			/* imageProps->setNumSamples(m_num);								// number of samples acquired on each scanline */
			/* imageProps->setImageResolution(double resolution);  			// the resolution of the scanConverted image */

			// Defines the type of transducer
			if (m_probeName == "Linear" || m_probeName == "CPLA12875" || m_probeName == "CPLA06475") {
				newProps->setTransducerType(USImageProperties::TransducerType::Linear);	
			} else if (m_probeName == "2D") {
				newProps->setTransducerType(USImageProperties::TransducerType::Bicurved);
			}


			// publish Rx Scanline parameters together with the RawData
			if(imageProps && imageProps->getScanlineInfo())
			{
				newProps->setScanlineInfo(imageProps->getScanlineInfo());
			}

			// geometrical beamformer-related settings
			newProps->setSpecificParameter("UsIntCepCc.numScanlines.x", numScanlines.x);
			newProps->setSpecificParameter("UsIntCepCc.numScanlines.y", numScanlines.y);
			newProps->setSpecificParameter("UsIntCepCc.rxScanlines.x", rxScanlines.x);
			newProps->setSpecificParameter("UsIntCepCc.rxScanlines.y", rxScanlines.y);
			newProps->setSpecificParameter("UsIntCepCc.txSteeringAngle.x", steeringAngle.x);
			newProps->setSpecificParameter("UsIntCepCc.txSteeringAngle.y", steeringAngle.y);
			newProps->setSpecificParameter("UsIntCepCc.txSectorAngle.x", sectorAngle.x);
			newProps->setSpecificParameter("UsIntCepCc.txSectorAngle.y", sectorAngle.y);

			newProps->setSpecificParameter("UsIntCepCc.apertureSize.x", apertureSize.x);
			newProps->setSpecificParameter("UsIntCepCc.apertureSize.y", apertureSize.y);
			newProps->setSpecificParameter("UsIntCepCc.txApertureSize.x", txApertureSize.x);
			newProps->setSpecificParameter("UsIntCepCc.txApertureSize.y", txApertureSize.y);
			newProps->setSpecificParameter("UsIntCepCc.txFocusActive", bf->getTxFocusActive());
			newProps->setSpecificParameter("UsIntCepCc.txFocusDepth", bf->getTxFocusDepth());
			newProps->setSpecificParameter("UsIntCepCc.txFocusWidth", bf->getTxFocusWidth());
			newProps->setSpecificParameter("UsIntCepCc.txCorrectMatchingLayers", bf->getTxCorrectMatchingLayers());
			newProps->setSpecificParameter("UsIntCepCc.numSamplesRecon", bf->getNumDepths());
			newProps->setSpecificParameter("UsIntCepCc.scanType", bf->getScanType());
			newProps->setSpecificParameter("UsIntCepCc.rxModeActive", bf->getRxModeActive());



			// setting specific for beam ensemble transmit, not handled not within beamformer
			newProps->setSpecificParameter("UsIntCepCc.txFrequency",m_beamEnsembleTxParameters.at(numSeq).txFrequency);
			newProps->setSpecificParameter("UsIntCepCc.txPrf", m_beamEnsembleTxParameters.at(numSeq).txPrf);
			newProps->setSpecificParameter("UsIntCepCc.txRepeatFiring", m_beamEnsembleTxParameters.at(numSeq).txRepeatFiring);
			newProps->setSpecificParameter("UsIntCepCc.txVoltage", m_beamEnsembleTxParameters.at(numSeq).txVoltage);
			newProps->setSpecificParameter("UsIntCepCc.txPulseType", m_beamEnsembleTxParameters.at(numSeq).txPulseType);
			newProps->setSpecificParameter("UsIntCepCc.txPulseInversion", m_beamEnsembleTxParameters.at(numSeq).txPulseInversion);


			newProps->setSpecificParameter("UsIntCepCc.txNumCyclesCephasonics",  m_beamEnsembleTxParameters.at(numSeq).txNumCyclesCephasonics);
			newProps->setSpecificParameter("UsIntCepCc.txNumCyclesManual", m_beamEnsembleTxParameters.at(numSeq).txNumCyclesManual);


			//publish system-wide parameter settings to properties object
			newProps->setSpecificParameter("UsIntCepCc.systemTxClock", m_systemTxClock);
			newProps->setSpecificParameter("UsIntCepCc.probeName", m_probeName);
			newProps->setSpecificParameter("UsIntCepCc.startDepth", m_startDepth);
			newProps->setSpecificParameter("UsIntCepCc.endDepth", m_endDepth);
			newProps->setSpecificParameter("UsIntCepCc.processorMeasureThroughput", m_processorMeasureThroughput);
			newProps->setSpecificParameter("UsIntCepCc.speedOfSound", m_speedOfSound);

			newProps->setSpecificParameter("UsIntCepCc.tgc", m_vgaGain);
			newProps->setSpecificParameter("UsIntCepCc.decimation", m_decimation);
			newProps->setSpecificParameter("UsIntCepCc.decimationFilterBypass", m_decimationFilterBypass);
			newProps->setSpecificParameter("UsIntCepCc.antiAliasingFilterFrequency", m_antiAliasingFilterFrequency);
			newProps->setSpecificParameter("UsIntCepCc.highPassFilterBypass", m_highPassFilterBypass);
			newProps->setSpecificParameter("UsIntCepCc.highPassFilterFrequency", m_highPassFilterFrequency);
			newProps->setSpecificParameter("UsIntCepCc.lowNoiseAmplifierGain", m_lowNoiseAmplifierGain);
			newProps->setSpecificParameter("UsIntCepCc.inputImpedance", m_inputImpedance);

			m_pSequencer->setUSImgProperties(numSeq, newProps);
		}
	}





	void UsIntCephasonicsCc::initializeDevice()
	{
		checkOptions();	
	
		m_cUSEngine = unique_ptr<USEngine>(new USEngine(*m_cPlatformHandle));
		m_cUSEngine->stop();
		m_cUSEngine->setBlocking(false);

		//Step 2 ----------------- "Create Scan Definition"
		setupScan();
		setupRxCopying();

		//Step 3 ----------------- "Create Ultrasound Engine Thread"
		//create the data processor that later handles the data
		m_pDataProcessor = unique_ptr<UsIntCephasonicsCcProc>(
			new UsIntCephasonicsCcProc(*m_cPlatformHandle, this)
			);
		if(m_processorMeasureThroughput)
		{
			m_pDataProcessor->setMeasureThroughput(true, 50000);
		}
		//Create execution thread to run USEngine
		m_runEngineThread = thread([this]() {
			//The run function of USEngine starts its internal state machine that will run infinitely
			//until the USEngine::teardown() function is called or a fatal exception.
			m_cUSEngine->run(*m_pDataProcessor);

			//This thread will only return null on teardown of USEngine.
		});

		std::this_thread::sleep_for (std::chrono::seconds(2));
		logging::log_log("USEngine: initialized");

		// After system startup we can get the actual voltages -> verify everthing is set correctly
		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			bool isUniPolar = (BeamEnsembleTxParameters::Unipolar == m_beamEnsembleTxParameters.at(numSeq).txPulseType);
			checkVoltageSetting(m_pFrameDefs.at(numSeq), m_beamEnsembleTxParameters.at(numSeq).txVoltage, isUniPolar);
		}


		m_ready = true;
	}
	
	std::vector<PulseVal> UsIntCephasonicsCc::createWeightedWaveform(
		const BeamEnsembleTxParameters& txParams, size_t numTotalEntries, float weight, uint8_t csTxOversample)
	{
		//Creating TX single pulse - The wave pulse is constructed at 4x the system clock.
		//Assuming the system clock is 40MHz, then below is an even duty cycle
		//NOTE:  Leading, trailing, and middle ground (for bipolar pulses) 
		//points are required in order to make a proper wave.
		//Calculation of num_pules to make frequency of transmit pulse is based on _txFreq, set by -f option

		//
		//The Wave Frequency is given by TxFrequency = SysClock*4/ (Number of pos pulses + Number of neg pulses)
		// NumPulseVals = SysClock*4 / TxFrequency
		// E.G: 2.5MHz Transmit Pulse Signal = 40*4/(32+32)
		double pulseLength = static_cast<double>(m_systemTxClock)*1e6*csTxOversample / (txParams.txFrequency*1e6);

		// Bipolar pulse with positive and negative half pulses (or the inverse)
		double pulseQuarterLength = pulseLength/4.0;
		double pulseQuarterLengthWeighted = weight*pulseQuarterLength;

		// set desired pulsing values depending on pulse inversion.
		auto pulsingValueLeft = (txParams.txPulseInversion == true) ? NEGV0 : POSV0;
		auto pulsingValueRight = (txParams.txPulseInversion == true) ? POSV0 : NEGV0;

		// target container for wave table
		vector<PulseVal> waveDef = vector<PulseVal> (ceil(pulseQuarterLength*4 + 3) * txParams.txNumCyclesManual, GND);

		for(size_t cycleIdx = 0; cycleIdx < txParams.txNumCyclesManual; cycleIdx++)
		{
			//Points to element with Leading Ground
			size_t firstIdx = cycleIdx*(pulseQuarterLength*4 + 3);

			//Points to location of left peak (+1 because of leading ground)
			double leftPeak = firstIdx + 1 + pulseQuarterLength ;
			for (size_t i = round(leftPeak-pulseQuarterLengthWeighted); 
				i < round(leftPeak+pulseQuarterLengthWeighted); i++)
			{
				waveDef[i] = pulsingValueLeft;
			}

			// Unipolar pulsing is finished here, for bipolar pulsing, add the second half cycle
			if (BeamEnsembleTxParameters::Bipolar == txParams.txPulseType)
			{
				// Points to location of right peak (+2 because of leading and mid ground)
				double rightPeak = firstIdx + 2 + (3*pulseQuarterLength);
				for (size_t i = round(rightPeak-pulseQuarterLengthWeighted); 
					i < round(rightPeak+pulseQuarterLengthWeighted); i++)
				{
					waveDef[i] = pulsingValueRight;
				}
			}		
		}

		return waveDef;
	}

	void UsIntCephasonicsCc::startAcquisition()
	{
		m_cUSEngine->start();
	}

	void UsIntCephasonicsCc::stopAcquisition()
	{
		if(m_cUSEngine)
		{
			m_cUSEngine->stop();       //Stop USEngine
		}
	}


	void UsIntCephasonicsCc::layoutChanged(ImageLayout & layout)
	{
		log_error("UsIntCephasonicsCc: layoutChanged was called NOT SUPPOSED TO HAPPEN.");
	}


	void UsIntCephasonicsCc::configurationDictionaryChanged(const ConfigurationDictionary& newConfig)
	{
		// check if number of beam sequences has changed in the new configuration
		// if it did, update the value ranges for each sequence
		size_t numBeamSequences = newConfig.get<uint32_t>("sequenceNumFrames", 1);
		if (m_numBeamSequences != numBeamSequences)
		{
			logging::log_log("UsIntCephasonicsCc: New number of beam sequences ", numBeamSequences, ", was formerly ", m_numBeamSequences);
			size_t oldNumBeamSequences = m_numBeamSequences;
			m_numBeamSequences = numBeamSequences;
			setBeamSequenceValueRange(oldNumBeamSequences);

			// create new sequencer
			m_pSequencer = unique_ptr<Sequencer>(new Sequencer(m_numBeamSequences));
			m_beamEnsembleTxParameters.resize(m_numBeamSequences);
		}
	}

	// change of multiple values in configuration result in a major (and re-initialization) update of all values
	void UsIntCephasonicsCc::configurationChanged()
	{
		// update systemwide configuration values
		m_systemTxClock = m_configurationDictionary.get<uint32_t>("systemTxClock");
		m_probeName = m_configurationDictionary.get<string>("probeName");
		m_startDepth = m_configurationDictionary.get<double>("startDepth");
		m_endDepth = m_configurationDictionary.get<double>("endDepth");
		m_processorMeasureThroughput= m_configurationDictionary.get<bool>("measureThroughput");
		m_speedOfSound = m_configurationDictionary.get<double>("speedOfSound");


		// support for test mock data
		m_writeMockData = m_configurationDictionary.get<bool>("writeMockData");
		m_mockDataFilename = m_configurationDictionary.get<string>("mockDataFilename");

		// iterate over all defined beam sequences, each beam-sequ defines one USImageProperties object
		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq);
			std::string seqIdApp = getBeamSequenceApp(m_numBeamSequences,numSeq);
			
			// scan or image-specific configuration values
			std::string scanType = m_configurationDictionary.get<std::string>(seqIdApp+"scanType");
			bf->setScanType(scanType);

			bf->setRxModeActive(m_configurationDictionary.get<bool>(seqIdApp+"rxModeActive"));
			bf->setTxFocusActive(m_configurationDictionary.get<bool>(seqIdApp+"txFocusActive"));
			bf->setTxFocusDepth(m_configurationDictionary.get<double>(seqIdApp+"txFocusDepth"));
			bf->setRxFocusDepth(m_configurationDictionary.get<double>(seqIdApp+"txFocusDepth")); // currently rx and tx focus are the same
			bf->setTxFocusWidth(m_configurationDictionary.get<double>(seqIdApp+"txFocusWidth"));
			bf->setTxCorrectMatchingLayers(m_configurationDictionary.get<bool>(seqIdApp+"txCorrectMatchingLayers"));
			bf->setNumDepths(m_configurationDictionary.get<uint32_t>(seqIdApp+"numSamplesRecon"));

			bf->setSpeedOfSound(m_speedOfSound);
			bf->setDepth(m_endDepth);

			vec2s numScanlines;
			numScanlines.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"numScanlinesX");
			numScanlines.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"numScanlinesY");
			bf->setNumTxScanlines(numScanlines);

			vec2s rxScanlinesSubdivision;
			rxScanlinesSubdivision.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"rxScanlineSubdivisionX");
			rxScanlinesSubdivision.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"rxScanlineSubdivisionY");
			bf->setRxScanlineSubdivision( rxScanlinesSubdivision );

			vec2 sectorAngle;
			sectorAngle.x = m_configurationDictionary.get<double>(seqIdApp+"txSectorAngleX");
			sectorAngle.y = m_configurationDictionary.get<double>(seqIdApp+"txSectorAngleY");
			bf->setTxSectorAngle(sectorAngle);

			vec2 steeringAngle;
			steeringAngle.x = m_configurationDictionary.get<double>(seqIdApp+"txSteeringAngleX");
			steeringAngle.y = m_configurationDictionary.get<double>(seqIdApp+"txSteeringAngleY");
			bf->setTxSteeringAngle(steeringAngle);

			vec2s apertureSize;
			apertureSize.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"apertureSizeX");
			apertureSize.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"apertureSizeY");
			bf->setMaxApertureSize(apertureSize);

			vec2s txApertureSize;
			txApertureSize.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"txApertureSizeX");
			txApertureSize.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"txApertureSizeY");
			bf->setTxMaxApertureSize(txApertureSize);
			
			string windowType = m_configurationDictionary.get<string>(seqIdApp+"txWindowType");
			bf->setTxWindowType(windowType);

			double winParam = m_configurationDictionary.get<double>("txWindowParameter");
			bf->setWindowParameter(winParam);

			// ensemble-specific parameters (valid for a whole image irrespective of whether it is linear, phased, planewave, or push)
			m_beamEnsembleTxParameters.at(numSeq).txPrf = m_configurationDictionary.get<double>(seqIdApp+"txPulseRepetitionFrequency");
			m_beamEnsembleTxParameters.at(numSeq).txRepeatFiring = m_configurationDictionary.get<uint32_t>(seqIdApp+"txPulseRepeatFiring");
			m_beamEnsembleTxParameters.at(numSeq).txVoltage = m_configurationDictionary.get<double>(seqIdApp+"txVoltage");

			std::string pulseType = m_configurationDictionary.get<string>(seqIdApp+"txPulseType");
			if ("unipolar" == pulseType)
			{
				m_beamEnsembleTxParameters.at(numSeq).txPulseType = BeamEnsembleTxParameters::Unipolar;
			}
			else if ("bipolar" == pulseType)
			{
				m_beamEnsembleTxParameters.at(numSeq).txPulseType = BeamEnsembleTxParameters::Bipolar;
			}
			else {
				logging::log_warn("UsIntCephasonicsCc: : Incorrect pulse type set, defaulting to Bipolar transmit pulse.");
			}
			
			m_beamEnsembleTxParameters.at(numSeq).txPulseInversion = m_configurationDictionary.get<bool>(seqIdApp+"txPulseInversion");
			m_beamEnsembleTxParameters.at(numSeq).txFrequency = m_configurationDictionary.get<double>(seqIdApp+"txFrequency");
			m_beamEnsembleTxParameters.at(numSeq).txDutyCycle = m_configurationDictionary.get<double>(seqIdApp+"txDutyCycle");
			m_beamEnsembleTxParameters.at(numSeq).txNumCyclesCephasonics = m_configurationDictionary.get<uint32_t>(seqIdApp+"txNumCyclesCephasonics");
			if (m_beamEnsembleTxParameters.at(numSeq).txNumCyclesCephasonics > 20)
			{
				logging::log_warn("UsIntCephasonicsCc: : Selected more than 20 cycles for pulse - Too long pulsing can damage hardware permanently!");
			}

			m_beamEnsembleTxParameters.at(numSeq).txNumCyclesManual = m_configurationDictionary.get<uint32_t>(seqIdApp+"txNumCyclesManual");
		}
		
		readVgaSettings();
		m_decimation = m_configurationDictionary.get<uint32_t>("decimation");
		m_decimationFilterBypass = m_configurationDictionary.get<bool>("decimationFilterBypass");
		m_antiAliasingFilterFrequency = m_configurationDictionary.get<double>("antiAliasingFilterFrequency");
		m_highPassFilterBypass = m_configurationDictionary.get<bool>("highPassFilterBypass");
		m_highPassFilterFrequency = m_configurationDictionary.get<double>("highPassFilterFrequency");
		m_lowNoiseAmplifierGain = m_configurationDictionary.get<double>("lowNoiseAmplifierGain");
		m_inputImpedance = m_configurationDictionary.get<double>("inputImpedance");

		updateTransducer();

		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			
			std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq);

			// (re)compute the internal TX parameters for a beamformer if any parameter changed
			if (!bf->isReady())
			{
				bf->computeTxParameters();
			}
		}

		updateImageProperties();
	}

	void UsIntCephasonicsCc::configurationEntryChanged(const std::string & configKey)
	{
		lock_guard<mutex> lock(m_objectMutex);

		// settings which can be changed online
		if(m_ready)
		{
			if(string("tgc").compare(0, 7, configKey))
			{
				readVgaSettings();
				applyVgaSettings();
			}
		}

		//these properties require large reconfigurations. for now allow them only before initialization
		if(!m_ready)
		{
			// global (system-wide) settings

			if(configKey == "endDepth")
			{
				m_endDepth = m_configurationDictionary.get<double>("endDepth");
			}
			if(configKey == "speedOfSound")
			{
				m_speedOfSound = m_configurationDictionary.get<double>("speedOfSound");
			}
			if(configKey == "decimation")
			{
				m_decimation = m_configurationDictionary.get<uint32_t>("decimation");
			}
			if(configKey == "decimationFilterBypass")
			{
				m_decimationFilterBypass = m_configurationDictionary.get<bool>("decimationFilterBypass");
			}
			if(configKey == "antiAliasingFilterFrequency")
			{
				m_antiAliasingFilterFrequency = m_configurationDictionary.get<double>("antiAliasingFilterFrequency");
			}
			if(configKey == "highPassFilterBypass")
			{
				m_highPassFilterBypass = m_configurationDictionary.get<bool>("highPassFilterBypass");
			}
			if(configKey == "highPassFilterFrequency")
			{
				m_highPassFilterFrequency = m_configurationDictionary.get<double>("highPassFilterFrequency");
			}
			if(configKey == "lowNoiseAmplifierGain")
			{
				m_lowNoiseAmplifierGain = m_configurationDictionary.get<double>("lowNoiseAmplifierGain");
			}
			if(configKey == "inputImpedance")
			{
				m_inputImpedance = m_configurationDictionary.get<double>("inputImpedance");
			}
			if(configKey == "measureThroughput")
			{
				m_processorMeasureThroughput= m_configurationDictionary.get<bool>("measureThroughput");
			}



			// local settings (per firing)
			for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
			{
				std::string seqIdApp = getBeamSequenceApp(m_numBeamSequences,numSeq);
				std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq);

				if(configKey == seqIdApp+"numScanlinesX" || configKey == seqIdApp+"numScanlinesY")
				{
					vec2s numScanlines;
					numScanlines.x = m_configurationDictionary.get<size_t>(seqIdApp+"numScanlinesX");
					numScanlines.y = m_configurationDictionary.get<size_t>(seqIdApp+"numScanlinesY");
					bf->setNumTxScanlines( numScanlines );
				}
				if(configKey == seqIdApp+"rxScanlineSubdivisionX" || configKey == seqIdApp+"rxScanlineSubdivisionY")
				{
					vec2s rxScanlinesSubdivision;
					rxScanlinesSubdivision.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"rxScanlineSubdivisionX");
					rxScanlinesSubdivision.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"rxScanlineSubdivisionY");
					bf->setRxScanlineSubdivision( rxScanlinesSubdivision );
				}

				if(configKey == seqIdApp+"txSectorAngleX" || configKey == seqIdApp+"txSectorAngleY")
				{
					vec2 sectorAngle;
					sectorAngle.x = m_configurationDictionary.get<double>(seqIdApp+"txSectorAngleX");
					sectorAngle.y = m_configurationDictionary.get<double>(seqIdApp+"txSectorAngleY");
					bf->setTxSectorAngle( sectorAngle );
				}
				if(configKey == seqIdApp+"txSteeringAngleX" || configKey == seqIdApp+"txSteeringAngleY")
				{
					vec2 steerAngle;
					steerAngle.x = m_configurationDictionary.get<double>(seqIdApp+"txSteeringAngleX");
					steerAngle.y = m_configurationDictionary.get<double>(seqIdApp+"txSteeringAngleY");
					bf->setTxSteeringAngle( steerAngle );
				}
				if(configKey == seqIdApp+"apertureSizeX" || configKey == seqIdApp+"apertureSizeY")
				{
					vec2s apertureSize;
					apertureSize.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"apertureSizeX");
					apertureSize.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"apertureSizeY");
					bf->setMaxApertureSize( apertureSize );
				}
				if(configKey == seqIdApp+"txApertureSizeX" || configKey == seqIdApp+"txApertureSizeY")
				{
					vec2s txApertureSize;
					txApertureSize.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"txApertureSizeX");
					txApertureSize.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"txApertureSizeY");
					bf->setTxMaxApertureSize( txApertureSize );
				}
				if(configKey == seqIdApp+"txFocusActive")
				{
					bf->setTxFocusActive(m_configurationDictionary.get<bool>(seqIdApp+"txFocusActive"));
				}
				if(configKey == seqIdApp+"txFocusDepth")
				{
					bf->setTxFocusDepth(m_configurationDictionary.get<double>(seqIdApp+"txFocusDepth"));
				}
				if(configKey == seqIdApp+"txFocusWidth")
				{
					bf->setTxFocusWidth(m_configurationDictionary.get<double>(seqIdApp+"txFocusWidth"));
				}
				if(configKey == seqIdApp+"txCorrectMatchingLayers")
				{
					bf->setTxCorrectMatchingLayers(m_configurationDictionary.get<bool>(seqIdApp+"txCorrectMatchingLayers"));
				}
				if(configKey == seqIdApp+"numSamplesRecon")
				{
					bf->setNumDepths(m_configurationDictionary.get<uint32_t>(seqIdApp+"numSamplesRecon"));
				}
			

				// beam ensemble specific transmit values
				if (configKey == seqIdApp+"txVoltage")
				{
					m_beamEnsembleTxParameters.at(numSeq).txVoltage = m_configurationDictionary.get<double>(seqIdApp+"txVoltage");
				}

				if (configKey == seqIdApp+"txPulseType")
				{
					string pulseType = m_configurationDictionary.get<string>(seqIdApp+"txPulseType");
					if ("unipolar" == pulseType)
					{
						m_beamEnsembleTxParameters.at(numSeq).txPulseType = BeamEnsembleTxParameters::Unipolar;
					}
					else if ("bipolar" == pulseType)
					{
						m_beamEnsembleTxParameters.at(numSeq).txPulseType = BeamEnsembleTxParameters::Bipolar;
					}
					else {
						logging::log_warn("UsIntCephasonicsCc: Incorrect pulse type set, defaulting to Bipolar transmit pulse.");
						m_beamEnsembleTxParameters.at(numSeq).txPulseType = BeamEnsembleTxParameters::Bipolar;
					}
				}
				if (configKey == seqIdApp+"txPulseInversion")
				{
					m_beamEnsembleTxParameters.at(numSeq).txPulseInversion = m_configurationDictionary.get<bool>(seqIdApp+"txPulseInversion");
				}

				if(configKey == seqIdApp+"txDutyCycle")
				{
					// TODO
				}
				if(configKey == seqIdApp+"txFrequency")
				{
					m_beamEnsembleTxParameters.at(numSeq).txFrequency = m_configurationDictionary.get<double>(seqIdApp+"txFrequency");
				}
				if(configKey == seqIdApp+"txPulseRepetitionFrequency")
				{
					m_beamEnsembleTxParameters.at(numSeq).txPrf = m_configurationDictionary.get<double>(seqIdApp+"txPulseRepetitionFrequency");
				}
				if(configKey == seqIdApp+"txPulseRepeatFiring")
				{
					m_beamEnsembleTxParameters.at(numSeq).txRepeatFiring = m_configurationDictionary.get<uint32_t>(seqIdApp+"txPulseRepeatFiring");
				}	
				if(configKey == seqIdApp+"txNumCyclesCephasonics")
				{
					m_beamEnsembleTxParameters.at(numSeq).txNumCyclesCephasonics = m_configurationDictionary.get<uint32_t>(seqIdApp+"txNumCyclesCephasonics");
				}
				if(configKey == seqIdApp+"txNumCyclesManual")
				{
					m_beamEnsembleTxParameters.at(numSeq).txNumCyclesManual = m_configurationDictionary.get<uint32_t>(seqIdApp+"txNumCyclesManual");
				}

				// update bf internal parameters if a relevant parameter has changed
				if (!bf->isReady())
				{
					bf->computeTxParameters();
				}
			}

		}
		updateImageProperties();
	}

	void UsIntCephasonicsCc::readVgaSettings()
	{
		m_vgaGain[0] = m_configurationDictionary.get<double>("tgc0");
		m_vgaGain[1] = m_configurationDictionary.get<double>("tgc1");
		m_vgaGain[2] = m_configurationDictionary.get<double>("tgc2");
		m_vgaGain[3] = m_configurationDictionary.get<double>("tgc3");
		m_vgaGain[4] = m_configurationDictionary.get<double>("tgc4");
		m_vgaGain[5] = m_configurationDictionary.get<double>("tgc5");
		m_vgaGain[6] = m_configurationDictionary.get<double>("tgc6");
		m_vgaGain[7] = m_configurationDictionary.get<double>("tgc7");
		m_vgaGain[8] = m_configurationDictionary.get<double>("tgc8");
		m_vgaGain[9] = m_configurationDictionary.get<double>("tgc9");
	}

	void UsIntCephasonicsCc::applyVgaSettings()
	{
		//VGA gain controlled by -v flag
		vector < TGC_Profile *> profiles;
		//Adding TGC_Profile i.e. Depth and gain to the profiles vector;
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.1, m_vgaGain[0]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.2, m_vgaGain[1]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.3, m_vgaGain[2]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.4, m_vgaGain[3]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.5, m_vgaGain[4]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.6, m_vgaGain[5]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.7, m_vgaGain[6]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.8, m_vgaGain[7]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 0.9, m_vgaGain[8]));
		profiles.push_back(new TGC_Profile(m_endDepth/1000 * 1.0, m_vgaGain[9]));

		// all TGC settings are equal for framedefs within a sequence
		for (auto it : m_pFrameDefs)
		{
			it->update(SetTGC(profiles));
		}
	}

	// apply new voltage 
	void UsIntCephasonicsCc::applyVoltageSetting(const cs::FrameDef* pFrameDef, double newVoltage, bool isUniPolar, bool noCheck)
	{

		// consider that cephasonics considers Vpp always bipolar, thus double unipolar target voltage
		double setVoltage = newVoltage * (isUniPolar ? 2.0 : 1.0);

		pFrameDef->update(SetTransmitVoltage(setVoltage));

		double voltage;
		pFrameDef->update(GetTransmitVoltage(voltage));

		if(! noCheck)
		{
			checkVoltageSetting(pFrameDef, newVoltage, isUniPolar);
		}
	}

	// check new voltage setting and log if not successful
	void UsIntCephasonicsCc::checkVoltageSetting(const FrameDef* pFrameDef, double targetVoltage, bool isUniPolar )
	{
		double voltage;
		pFrameDef->update(GetTransmitVoltage(voltage));

		// consider that cephasonics considers Vpp always bipolar, thus double unipolar target voltage
		double setVoltage = targetVoltage * (isUniPolar ? 2.0 : 1.0);
		
		if(voltage != setVoltage)
		{
			//retry
			applyVoltageSetting(pFrameDef, targetVoltage, isUniPolar, true);
			pFrameDef->update(GetTransmitVoltage(voltage));
		}

		if(voltage > setVoltage)
		{
			logging::log_warn("UsIntCephasonics: Transmit voltage requested: ", setVoltage, "V, system reported: ", voltage, "V - Please proceed with care.");
			//CS_THROW("Applied voltage is higher than requested. Emergency stop!");
		}
		if(voltage < setVoltage)
		{
			logging::log_warn("UsIntCephasonics: Transmit voltage requested: ", setVoltage, "V, system reported: ", voltage, "V were set - Image quality might be deteriorated.");
		}
	}


	// A scan consist of the overall scanning sequence and can consit of multiple frames to be acquire sequentially/periodically
	void UsIntCephasonicsCc::setupScan()
	{
		//create new transducer
		updateTransducer();
		setupCsProbe();

		bool swModeEn = true;                      //need to set this to true when making fully-custom beams
		m_pScan = &ScanDef::createScanDef(*m_cPlatformHandle,
										*m_pProbe,
										swModeEn,
										m_speedOfSound,
										RX_BUFFER_SZ); //ignored in CHCAP mode

		//controlled by the -C flag, default is 40MHz or can be 20MHz
		if (m_systemTxClock == 20)
		{
		  m_pScan->update(SetClock(RX, INP_CLK1, false, 2, 40000000));
		  m_pScan->update(SetClock(TX, INP_CLK1, false, 1, 20000000));
		  m_systemRxClock = 40;
		}
		else
		{
		  m_pScan->update(SetClock(RX, INP_CLK1, false, 1, 40000000));
		  m_pScan->update(SetClock(TX, INP_CLK1, false, 1, 40000000));
		  m_systemRxClock = 40;
		}

		// sets up the transducer, beamformer and creates the FrameDef
		createSequence();

		//apply the default VGA seetings (used for TGC)
		applyVgaSettings();

		//Input impedance controlled by -i flag
		m_pScan->update(SetLNA(m_lowNoiseAmplifierGain, m_inputImpedance));

		//controlled by -a
		m_pScan->update(SetAAF(m_antiAliasingFilterFrequency * 1e6));

		//Decimation filter
		if(m_decimation == 1 && m_decimationFilterBypass == false)
		{
			log_error("UsIntCephasonicsCc: Decimation FIR filtering must be bypassed when decimation is 1! Activating bypass.");
			m_decimationFilterBypass = true;
		}

		if (m_decimationFilterBypass)
		{
		  m_pScan->update(SetDecimFiltBypass(true));
		}

		m_pScan->update(SetDecimFiltDecimation(m_decimation));
		if (m_decimation != 1)
		{
			std::map<uint16, std::vector<std::vector<double> > > coeffs_map;
			std::vector<std::vector<double> > coeffs2(32);
			std::vector<std::vector<double> > coeffs4(32);
			std::vector<std::vector<double> > coeffs8(32);
			std::vector<std::vector<double> > coeffs16(32);
			std::vector<std::vector<double> > coeffs32(32);
			for (uint32 x = 0; x < 32; x++)
			{
				coeffs2[x].assign(myCoeffs2[x], myCoeffs2[x]+2*8);
				coeffs4[x].assign(myCoeffs4[x], myCoeffs4[x]+4*8);
				coeffs8[x].assign(myCoeffs8[x], myCoeffs8[x]+8*8);
				coeffs16[x].assign(myCoeffs16[x], myCoeffs16[x]+16*8);
				coeffs32[x].assign(myCoeffs32[x], myCoeffs32[x]+32*8);
			}
			coeffs_map[2]  = coeffs2;
			coeffs_map[4]  = coeffs4;
			coeffs_map[8]  = coeffs8;
			coeffs_map[16] = coeffs16;
			coeffs_map[32] = coeffs32;

			// decimation identical for all frames in scan right now
			// TODO: change this to frame-specific decimation
			m_pScan->update(SetDecimFiltCoeffs(coeffs_map.at(m_decimation), 0));
		}

		m_pScan->update(SetTimeout(500,true));

		m_pScan->update(SetHPF(m_highPassFilterFrequency*1e6, m_highPassFilterBypass));

		//TODO add this later iff neccessary
		/*
		m_pScan->update(SetTXChannel(m_txChannelMap));
		m_pScan->update(SetRXChannel(m_rxChannelMap));
		if (_txChannel != "")
		{
		  txChannelVect = createChannelVector(_txChannel,pC.SYS_LICENSED_CHANNELS);
		  //controlled by -t flag
		  m_pScan->update(SetTXChannel(txChannelVect));
		}

		if (_rxChannel != "")
		{
		  rxChannelVect = createChannelVector(_rxChannel,pC.SYS_LICENSED_CHANNELS);
		  //controlled by -r flag
		  m_pScan->update(SetRXChannel(rxChannelVect));
		}

		if (_muxElement != "")
		{
		  muxElementVect = createChannelVector(_muxElement,pC.SYS_LICENSED_CHANNELS*pC.MUX_RATIO);
		  //controlled by -m flag
		  m_pScan->update(SetMuxElement(muxElementVect));
		}

		if (_muxSwitch != "")
		{
		  muxSwitchVect = createChannelVector(_muxSwitch,pC.SYS_LICENSED_CHANNELS);
		  //controlled by -s flag
		  m_pScan->update(SetMuxSwitch(muxSwitchVect));
		}*/

		m_cUSEngine->setScanDef(*m_pScan);
	}


	void UsIntCephasonicsCc::createSequence()
	{
		
		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
	
			std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq);
			std::shared_ptr<USImageProperties> props = m_pSequencer->getUSImgProperties(numSeq);

			// push rx parameters to US properties
			props->setScanlineInfo(bf->getRxParameters());

			// Get the Tx scanline parameters to program the Hardware with them
			const std::vector<ScanlineTxParameters3D>* beamTxParams = bf->getTxParameters();
			bool disableRx = !bf->getRxModeActive();

			std::pair<size_t, const cs::FrameDef*> fdef = createFrame(beamTxParams, props, m_beamEnsembleTxParameters.at(numSeq), disableRx);
										
			// store framedef and add it to Cephasonics interface
			m_pFrameMap[fdef.first] = m_sequenceNumFrames;
			m_pFrameDefs.push_back(fdef.second);

			// add entry to receive map, which will be used to identify complete sequences and/or dropped frames
			m_sequenceFramesReceived.push_back(false);	
			m_sequenceNumFrames++;
		}
	}

	// a frame is defined as set of BeamEnsembles. Multiple frames can be put together into a sequence
	// SUPRA Frames have a 1:1 correspondene to a beamformer, i.e. a Beamformer is generating necessary info for a frame
	std::pair<size_t, const cs::FrameDef*> UsIntCephasonicsCc::createFrame(
		const std::vector<ScanlineTxParameters3D>* txBeamParams, 
		const std::shared_ptr<USImageProperties> imageProps, 
		const BeamEnsembleTxParameters& txEnsembleParams,
		const bool disableRx)
	{

		// publish Rx Scanline parameters together with the RawData
		// updateImageProperties();


		// scanlines
		vec2s numScanlines = imageProps->getScanlineLayout();

		vector<const BeamEnsembleDef*> beamEnsembles;
		for(auto txParams: *txBeamParams)
		{
			// create the beam ensemble for this txBeam
			const BeamEnsembleDef* ensembleDef = createBeamEnsembleFromScanlineTxParameter(txEnsembleParams, numScanlines, txParams);
			beamEnsembles.push_back(ensembleDef);
		}


		// Create the SubFrameDef
		uint32 npb = 1; 					//ignored if making full custom beams
		double txFreq = 2500000; 			//ignored if making full custom beams
		double focalDepth = 0.07;
		double txFstop = 3.0; 				//ignored if making full custom beams
		double rxFstop = 2.0; 				//ignored if making full custom beams
		double angle = 0; 					//ignored if making full custom beams
		double centerAngle = 0; 			//ignored if making full custom beams
		rxWindowType rxWindow = RECTANGULAR;//ignored if making full custom beams

		// create a new SubFrameDef and provide it with the vector of BeamEnsembleDefs
		const SubFrameDef* sf = &SubFrameDef::createSubFrameDef(*m_cPlatformHandle,
				LINEAR, //not used for fully custom beams beam creation
				numScanlines.x*numScanlines.y, npb, txFreq, txEnsembleParams.txNumCyclesCephasonics, m_startDepth/1000, m_endDepth/1000,
				focalDepth, txFstop, rxFstop, rxWindow, angle, centerAngle,
				beamEnsembles, -1, -1, false, disableRx);
		m_pSubframeDefs.push_back(sf);
		size_t subframeID = sf->getID();

		//chose power rail depending on selected voltage
		// TODO extract voltage rail as parameter setting from config
		PlatformCapabilities pC = USPlatformMgr::getPlatformCapabilities(*m_cPlatformHandle);
		railType rail = DEFAULT;
		logging::log_log("UsIntCephasonicsCc: Reported voltage range RailA: ", pC.RAILA_VOLTAGE_MIN, " - ", pC.RAILA_VOLTAGE_MAX, "V");
		logging::log_log("UsIntCephasonicsCc: Reported voltage range RailB: ", pC.RAILB_VOLTAGE_MIN, " - ", pC.RAILB_VOLTAGE_MAX, "V");
		
		bool isUniPolar = BeamEnsembleTxParameters::Unipolar == txEnsembleParams.txPulseType;
		double targetVoltage = txEnsembleParams.txVoltage * (isUniPolar ? 2.0 : 1.0);
		bool alternateRail = m_pSubframeDefs.size() > 1;

		if (targetVoltage <= pC.RAILB_VOLTAGE_MAX && targetVoltage >= pC.RAILB_VOLTAGE_MIN && !alternateRail)
		{
			// Beware that Rail B only reports 110V right now...weird
			// Rail B is 6 times stronger thatn Rail A, thus we use B per default
			rail = RAIL_B;
			logging::log_log("UsIntCephasonicsCc: Setting rail B");
		}
		else if(targetVoltage <= pC.RAILA_VOLTAGE_MAX && targetVoltage >= pC.RAILA_VOLTAGE_MIN)
		{
			rail = RAIL_A;
			logging::log_log("UsIntCephasonicsCc: Setting rail A");
		}
		else {
			CS_THROW("Voltage not supported. Emergency stop!");
		}


		const FrameDef* fdef = &FrameDef::createFrameDef(
				*m_cPlatformHandle,
				CHCAP,
				*sf,
				false, 		// reverse
				rail); 	// railType

		// publish the framedef to the Cephasocnis API
		// add frame to scan to allow proper handling of parent-child structure
		m_pScan->update(AddFrame(*fdef));

		fdef->update(SetPRF(txEnsembleParams.txPrf,txEnsembleParams.txRepeatFiring));

		//We cannot check the voltage right now, as the frameDef is not completely determined
		applyVoltageSetting(fdef, txEnsembleParams.txVoltage, isUniPolar);

		return std::pair<size_t, const cs::FrameDef*>(subframeID,fdef);
	}

	const BeamEnsembleDef* UsIntCephasonicsCc::createBeamEnsembleFromScanlineTxParameter(
		const BeamEnsembleTxParameters& txEnsembleParameters, 
		const vec2s numScanlines, 
		const ScanlineTxParameters3D& txParameters)
	{
		PlatformCapabilities pC = USPlatformMgr::getPlatformCapabilities(*m_cPlatformHandle);


		//Creating TX single pulse
		//The wave pulse is constructed at 4x the system clock.
		//Assuming the system clock is 40MHz, then below is an even duty cycle
		//32 positive, 32 negative pulse waveform.  NOTE:  The leading, trailing, and middle ground
		//points are required in order to make a proper wave.
		//The Wave Frequency is given by TXFREQ = SysClock*4/Number of pos pulses + Number of neg pulses
		//2.5MHz Transmit Pulse Signal = 40*4/(32+32)
		//Calculation of num_pules to make frequency of transmit pulse is based on _txFreq, set by -f option
		size_t pulseHalfLength = (
				static_cast<double>(m_systemTxClock)*1e6*
				pC.TX_OVERSAMPLE /
				(txEnsembleParameters.txFrequency*1e6)
				)/2 - 1;
				

		size_t numTotalEntries = (pulseHalfLength + 1)*2 * txEnsembleParameters.txNumCyclesManual + 1;

		//In the case where the total number of pulses used exceeds 247, this means that the wave table would be stored
		//to the advanced if long pulse memory.  This long pulse memory can not be delayed.  However, in a case where a lower
		//frequency is desired, which requires more than 247 pulse train, we instead use 247 pulse train in conjunction with
		//the tx frequency divider to achieve the desired transmit frequency.
		if (numTotalEntries > MAX_WAVE_TABLE_SHORT_PULSE_LEN) //in this case we have to use tx divisor to make the pulse
		{
			log_error("UsIntCephasonicsCc: pulses this long have not been tested yet");
			pulseHalfLength = MAX_WAVE_TABLE_SHORT_PULSE_LEN/2;
			numTotalEntries = (pulseHalfLength + 1)*2 * txEnsembleParameters.txNumCyclesManual + 1;
		}


		// create passive wave with equal length
		vector<PulseVal> myWaveDef_passive(numTotalEntries, GND);

		vec2s elementLayout = m_pTransducer->getElementLayout();
		// create tx map w.r.t. the muxed channels
		vector<bool> txMap(m_numMuxedChannels, false);
		// the transmit delay vector also has to be specified for all muxed channels
		vector<double> txDelays(m_numMuxedChannels, 0);
		vector<vector<PulseVal> > txWaves(m_numMuxedChannels, myWaveDef_passive);
		for (size_t activeElementIdxX = txParameters.firstActiveElementIndex.x; activeElementIdxX <= txParameters.lastActiveElementIndex.x; activeElementIdxX++)
		{
			for (size_t activeElementIdxY = txParameters.firstActiveElementIndex.y; activeElementIdxY <= txParameters.lastActiveElementIndex.y; activeElementIdxY++)
			{
				if(txParameters.elementMap[activeElementIdxX][activeElementIdxY]) //should be true all the time, except we explicitly exclude elements
				{
					size_t muxedChanIdx = m_probeElementsToMuxedChannelIndices[activeElementIdxX + elementLayout.x*activeElementIdxY];
					txMap[muxedChanIdx] = true;
				}
			}
		}

		for (size_t activeElementIdxX = txParameters.txAperture.begin.x; activeElementIdxX <= txParameters.txAperture.end.x; activeElementIdxX++)
		{
			for (size_t activeElementIdxY = txParameters.txAperture.begin.y; activeElementIdxY <= txParameters.txAperture.end.y; activeElementIdxY++)
			{
				if(txParameters.txElementMap[activeElementIdxX][activeElementIdxY])
				{
					size_t localElementIdxX = activeElementIdxX -txParameters.txAperture.begin.x;
					size_t localElementIdxY = activeElementIdxY -txParameters.txAperture.begin.y;
					size_t muxedChanIdx = m_probeElementsToMuxedChannelIndices[activeElementIdxX + elementLayout.x*activeElementIdxY];

					//TX Delays are given in units of 4* TX_CLOCK
					// -> convert from seconds to 4*TX_Clock
					txDelays[muxedChanIdx]  = max(txParameters.delays[localElementIdxX][localElementIdxY]*static_cast<double>(m_systemTxClock*4)*1e6, M_EPS); // from seconds to microseconds
					
					auto txWeight = txParameters.weights[localElementIdxX][localElementIdxY];
					txWaves[muxedChanIdx] = createWeightedWaveform(txEnsembleParameters, numTotalEntries, txWeight, pC.TX_OVERSAMPLE);
				}
				else {
					//			txWaves[elemIdx] = myWaveDef_passive;
				}
			}
		}

		const BeamDef* txBeam = &BeamDef::createTXBeamDef(
				*m_cPlatformHandle,
				txWaves,
				txDelays);

		//size_t txFiringID = txBeam->getID();

		//NOTE: RX Delay and Weight are not used in CHCAP mode
		uint16_t vector_entries = (numScanlines.x*numScanlines.y > pC.MAX_RX_BEAMS_SWMODE/2) ? 8 : 16;
		vector<vector<double> > rxDelayVector(m_numMuxedChannels, vector<double>(vector_entries,0));
		vector<vector<double> > rxWeightVector(m_numMuxedChannels, vector<double>(vector_entries,0));

		const BeamDef* rxBeam = &BeamDef::createRXBeamDef(
				*m_cPlatformHandle,
				rxDelayVector,
				rxWeightVector);

		//size_t rxBeamID = rxBeam->getID();

		const FiringDef* firing = &FiringDef::createFiringDef(*m_cPlatformHandle, *txBeam,
						vector<const BeamDef*>(1, rxBeam), txMap);
		//TODO use this version
		/*const FiringDef* firing = &FiringDef::createFiringDef(*m_cPlatformHandle, *txBeam,
						vector<const BeamDef*>(1, rxBeam), txMap, rxMap);*/
		const BeamEnsembleDef* beamEnsemble = &BeamEnsembleDef::createBeamEnsembleDef(
				*m_cPlatformHandle, 0.0, //PRF=0.0 sets to maximum PRF
				//override by option is handled by SetPRF command below
				1,//has to be set to one for 2D-scans
				*firing);
		
		//size_t beamEnsembleID = beamEnsemble->getID();
		return beamEnsemble;
	}

	void UsIntCephasonicsCc::setupRxCopying()
	{
		//remember which platforms need to be copied at all
		m_rxPlatformsToCopy.resize(m_numPlatforms, false);

		for (size_t muxedChannelIndex : m_probeElementsToMuxedChannelIndices)
		{
			size_t channelIndex = muxedChannelIndex % m_numChannelsTotal;
			size_t platformIndex = channelIndex / m_numChannelsPerPlatform;
			m_rxPlatformsToCopy[platformIndex] = true;
		}

		//calculate how many the offset of the copying (in terms of platforms)
		m_rxPlatformCopyOffset.resize(m_numPlatforms, 0);
		size_t platformsToCopy = 0;
		for(size_t platformIndex = 0; platformIndex < m_numPlatforms; platformIndex++)
		{
			if(m_rxPlatformsToCopy[platformIndex])
			{
				m_rxPlatformCopyOffset[platformIndex] = platformsToCopy;
				platformsToCopy++;
			}
		}
		m_rxNumPlatformsToCopy = platformsToCopy;
	}


	void UsIntCephasonicsCc::putData(uint16_t platformIndex, size_t frameIndex, size_t frameNumber, uint32_t numChannels, size_t numSamples, size_t numBeams, uint8_t* dataScrambled)
	{
		double timestamp = getCurrentTime();

		// if frameNumber changes, check whether all subframe data (data from each beamformer) has arrived, yet
		if (m_lastFrameNumber != frameNumber)
		{
			bool allSequenceFramesReceived = true;
			for (auto sequFrameReceived : m_sequenceFramesReceived)
			{
				if (!sequFrameReceived)
				{
					allSequenceFramesReceived = false;
					m_numDroppedFrames++;
				}
				else
				{
					m_numReceivedFrames++;
				}
			}

			m_numDroppedFrames++;
			m_lastFrameNumber = frameNumber;
			m_sequenceFramesReceived.assign(m_sequenceNumFrames, false);


			if (!allSequenceFramesReceived)
			{
				log_error("UsIntCephasonicsCc: frame dropped. Total number of dropped frames: " + std::to_string(m_numDroppedFrames));
			}
		}
		
		// mark sequence frame triggered
		m_sequenceFramesReceived[m_pFrameMap[frameIndex]] = true;	

		//return;

		static vector<bool> platformsReceived;
		static size_t sNumBeams = 0;
		static size_t sNumChannels = 0;
		static size_t sArraySize = 0;
		static shared_ptr<Container<int16_t> > pData = nullptr;

		if(platformsReceived.size() != m_numPlatforms)
		{
			platformsReceived.resize(m_numPlatforms, false);
		}
		if(numBeams != sNumBeams || numChannels != sNumChannels)
		{
			sNumBeams = numBeams;
			sNumChannels = numChannels;
			sArraySize = m_rxNumPlatformsToCopy*m_numChannelsPerPlatform * numSamples * sNumBeams;
			pData = make_shared<Container<int16_t> >(ContainerLocation::LocationGpu, ContainerFactory::getNextStream(), sArraySize);

			platformsReceived.assign(m_numPlatforms, false);
		}

		platformIndex = platformIndex % m_numPlatforms;
		size_t numBytesChannels = numChannels*12/8;

		if(m_rxPlatformsToCopy[platformIndex])
		{
			{
				auto deviceScrambled = unique_ptr<Container<uint8_t> >(new Container<uint8_t>(LocationGpu, pData->getStream(), numBytesChannels*numSamples*numBeams));
				cudaSafeCall(cudaMemcpyAsync(deviceScrambled->get(), dataScrambled, numBytesChannels*numSamples*numBeams*sizeof(uint8_t), cudaMemcpyHostToDevice, pData->getStream()));

				size_t platformOffset = m_rxPlatformCopyOffset[platformIndex];
				dim3 blockSize(16, 8);
				dim3 gridSize(
						static_cast<unsigned int>((numSamples + blockSize.x - 1) / blockSize.x),
						static_cast<unsigned int>((numBeams + blockSize.y - 1) / blockSize.y));
				copyUnscramble<<<gridSize, blockSize, 0, pData->getStream()>>>(
						numBeams,
						numSamples,
						numChannels,
						numBytesChannels,
						platformOffset,
						m_rxNumPlatformsToCopy,
						deviceScrambled->get(),
						pData->get());
				cudaSafeCall(cudaPeekAtLastError());
			}
			//copy all raw data
			/*for(size_t beam = 0; beam < numBeams; beam++)
			{
				for(size_t sample = 0; sample < numSamples; sample++)
				{
					for(size_t channel = 0; channel < (numChannels/4); channel ++)
					{
						size_t channelOut;
						if(channel*4 < 16)
						{
							channelOut = channel*4;
						}
						else if(channel*4 < 32)
						{
							channelOut = channel*4 + 16;
						}
						else if(channel*4 < 48)
						{
							channelOut = channel*4 - 16;
						}
						else {
							channelOut = channel*4;
						}
						CephasonicsRawData4Channels d;
						d.raw[0] = dataScrambled[ channel*6 +     sample*numBytesChannels + beam*numBytesChannels*numSamples];
						d.raw[1] = dataScrambled[ channel*6 + 1 + sample*numBytesChannels + beam*numBytesChannels*numSamples];
						d.raw[2] = dataScrambled[ channel*6 + 2 + sample*numBytesChannels + beam*numBytesChannels*numSamples];
						d.raw[3] = dataScrambled[ channel*6 + 3 + sample*numBytesChannels + beam*numBytesChannels*numSamples];
						d.raw[4] = dataScrambled[ channel*6 + 4 + sample*numBytesChannels + beam*numBytesChannels*numSamples];
						d.raw[5] = dataScrambled[ channel*6 + 5 + sample*numBytesChannels + beam*numBytesChannels*numSamples];

						int16_t o1 = d.ordered.c3;
						int16_t o2 = d.ordered.c2;
						int16_t o3 = d.ordered.c1;
						int16_t o4 = d.ordered.c0;

						pData->get()[sample + (channelOut + 0 + platformOffset*m_numChannelsPerPlatform)*numSamples +
													 beam*m_numChannelsPerPlatform*m_rxNumPlatformsToCopy*numSamples] = o1;
						pData->get()[sample + (channelOut + 1 + platformOffset*m_numChannelsPerPlatform)*numSamples +
																		 beam*m_numChannelsPerPlatform*m_rxNumPlatformsToCopy*numSamples] = o2;
						pData->get()[sample + (channelOut + 2 + platformOffset*m_numChannelsPerPlatform)*numSamples +
																		 beam*m_numChannelsPerPlatform*m_rxNumPlatformsToCopy*numSamples] = o3;
						pData->get()[sample + (channelOut + 3 + platformOffset*m_numChannelsPerPlatform)*numSamples +
																		 beam*m_numChannelsPerPlatform*m_rxNumPlatformsToCopy*numSamples] = o4;
					}
				}
			}*/
			platformsReceived[platformIndex] = true;

			bool allreceived = true;
			for (size_t platformToCheck = 0; platformToCheck < m_numPlatforms; platformToCheck++)
			{
				allreceived = allreceived && (platformsReceived[platformToCheck] || !m_rxPlatformsToCopy[platformToCheck]);
			}
			if(allreceived)
			{
				lock_guard<mutex> lock(m_objectMutex);
				m_callFrequency.measure();

				size_t linFrID = m_pFrameMap[frameIndex];

				//build filename
				/*std::stringstream filename;
				filename << "/mnt/data/ascii_test/rawData_copiedtogether.txt";
				writeAscii(filename.str(), pData.get(), sArraySize);*/

				std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(linFrID);
				std::shared_ptr<USImageProperties> imProps = m_pSequencer->getUSImgProperties(linFrID);

				// we received the data from all necessary platforms, now we can start the beamforming
				shared_ptr<USRawData> rawData = make_shared<USRawData>
					(numBeams,
					m_pTransducer->getNumElements(),
					m_pTransducer->getElementLayout(),
					m_rxNumPlatformsToCopy*m_numChannelsPerPlatform,
					numSamples,
					 m_systemRxClock * 1e6, //MHz to Hz
					pData,
					bf->getCurrentRxBeamformerParameters(),
					imProps,
					timestamp,
					timestamp);

				if(m_writeMockData && !m_mockDataWritten)
				{
					rawData->getRxBeamformerParameters()->writeMetaDataForMock(m_mockDataFilename, const_pointer_cast<const USRawData>(rawData));
					m_mockDataWritten = true;
				}

				pData = make_shared<Container<int16_t> >(ContainerLocation::LocationGpu, ContainerFactory::getNextStream(), sArraySize);
				platformsReceived.assign(m_numPlatforms, false);

				// switch outputs for sequences. I.e. first sequence transmitted on output port 0, second port 1, etc.
				// max ouptut ports is 2 for the moment.
				switch (linFrID)
				{
					case 0:	addData<0>(rawData);
					break;

					case 1: 
						addData<1>(rawData);
					break;
					
					default: addData<0>(rawData);
					break;
				}
			}
		}

		// double timeDiff = getCurrentTime() - timestamp;
		// printf("time: %lf\n", timeDiff);
	}

	bool UsIntCephasonicsCc::ready()
	{
		return m_ready;
	}

	void UsIntCephasonicsCc::freeze()
	{
		m_cUSEngine->stop();
	}

	void UsIntCephasonicsCc::unfreeze()
	{
		m_cUSEngine->start();
	}
}
