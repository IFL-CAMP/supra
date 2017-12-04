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

#include "UsIntCephasonicsBtccProc.h"
#include "UsIntCephasonicsBtcc.h"

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
#include "utilities/utility.h"
#include "utilities/CallFrequency.h"
#include "utilities/Logging.h"
#include "ContainerFactory.h"

namespace supra
{
	using namespace ::std;
	using namespace ::cs;
	using namespace logging;

	const map<uint16, vector<vector<double> > > UsIntCephasonicsBtcc::coeffs_map = UsIntCephasonicsBtcc::create_coeffs_map();
	bool UsIntCephasonicsBtcc::m_environSet = false;

	//TODO replace CS_THROW with something more intelligent

	//Step 1 ------------------ "Setup Platform"
	PlatformHandle* UsIntCephasonicsBtcc::setupPlatform()
	{
		PlatformHandle* discoveredPlatforms[USPlatformMgr::MAX_HANDLES];
		int numPlatforms;

		numPlatforms = USPlatformMgr::instance()->discoverPlatforms(discoveredPlatforms);
		if (numPlatforms < 1 || numPlatforms >(int)USPlatformMgr::MAX_HANDLES)
		{
			CS_ERROR_LOG("BTCC", "Number of platforms is out of range, numPlatforms=" << numPlatforms);
		}
		if (numPlatforms == 0)
		{
			CS_ERROR_LOG("BTCC", "No Platforms connected, numPlatforms=" << numPlatforms);
		}
		PlatformHandle* ph = discoveredPlatforms[0];

		//controlled by -n flag
		PlatformCapabilities pC = USPlatformMgr::getPlatformCapabilities(*ph);

		_numPlatforms = (1 + USPlatformMgr::instance()->getNumSlaves(*ph));
		printf("Number of connected platforms: %d\n", _numPlatforms);

		if (_numElem == 0)
		{
			NUM_EFF_ELEMENTS = pC.SYS_LICENSED_CHANNELS * pC.MUX_RATIO;
		}
		else
		{
			NUM_EFF_ELEMENTS = _numElem;
		}
		NUM_CUSTOM_BEAMS = NUM_EFF_ELEMENTS / _simuFireNumElem;

		printf("Licensed channels: %d\nMux ratio: %d\nNUM_EFF_ELEMENTS: %d\nNUM_CUSTOM_BEAMS: %d\nMAX_RX_BEAMS_SWMODE: %d\n",
			pC.SYS_LICENSED_CHANNELS,
			pC.MUX_RATIO,
			NUM_EFF_ELEMENTS,
			NUM_CUSTOM_BEAMS,
			pC.MAX_RX_BEAMS_SWMODE);

		if (NUM_EFF_ELEMENTS > pC.SYS_LICENSED_CHANNELS * pC.MUX_RATIO)
		{
			CS_THROW("More elements specified than possible muxed channels");
		}
		if (NUM_CUSTOM_BEAMS > pC.MAX_RX_BEAMS_SWMODE)
		{
			CS_THROW("More beams specified than possible, use -n option to limit the number of beams");
		}

		return ph;
	}

	void UsIntCephasonicsBtcc::setupProbe()
	{
		string probeName = "BlinkCC";        //Probe Name
		string probeSN = "Unknown";          //Probe Serial No
		uint16 mapping = _probeMapping;      //Probe Mapping, supplied with -M option, default = 0
		double minFreq = 2000000;            //Probe MinFrequency
		double maxFreq = 7000000;            //Probe MaxFrequency
		double maxVoltage = 120;             //Probe MaxVoltage
		double yLen = 0;                     //Probe YLen, not used currently
		uint32 xNumElem = NUM_EFF_ELEMENTS;  //Probe xNumElem, can be set by -n option, otherwise set to
											 //maximum elelments system can support
		double xLen = 2.53e-2;               //Probe Width is irrelevent for custom beam scans
		uint32 yNumElem = 1;                 //Probe yNumElem

		_probe = (Probe*)&LinearProbe::createProbe(*m_cPlatformHandle, probeSN, probeName,
			minFreq, maxFreq, maxVoltage, mapping,
			xLen, xNumElem, yLen, yNumElem);
	}

	void UsIntCephasonicsBtcc::setProgramOptions()
	{
		_maxDump = 3;			//MaxDumpFiles
		_prf = 0;				//PRF
		_numElem = 0;			//NumElements
		_simuFireNumElem = 3;   //SimuFireNumElements (default was 1, but we have to use 3 because of the muxing)
		_probeMapping = 0;		//ProbeMapping
		_firBypass = true;		//DecimFiltBypass (default was false)
		_HPFBypass = true; 		//HPFBypass (defualt was false)
		_clock = 40;			//SystemClock (MHz)
		_startDepth = 0;		//StartDepth (m)
		_endDepth = 0.05;		//EndDepth (m) (default was 0.01, but we were advised to 0.05)
		_tpEn = false;			//TestPattern
		_dec = 1;				//Decimation  (default was 16)
		_inpimp = 200;			//InputImpedance (ohm)
		_aaf = 15000000;		//AAF Cutoff Frequency (Hz)
		_hpf = 898000;			//HPF Cutoff Frequence (Hz)
		_numCycles = 1; 		//NumCycles
		_txFreq = 2500000; 		//Transmit Frequency (Hz)
		_fullAperture = false; 	//Transmit FullAperture
		_vgaGain = 20; 			//VGA gain (dB)
		_throughput = false; 	//measure throughput


		_decPresent = _dec != 0;
		_firBypassPresent = _firBypass;
		_HPFBypassPresent = _HPFBypass;
	}

	void UsIntCephasonicsBtcc::checkOptions()
	{
		if (!(_clock == 20 || _clock == 40))
		{
			CS_THROW("Clock can only be set to 20 MHz or 40 MHz.");
		}
		PlatformCapabilities pC = USPlatformMgr::getPlatformCapabilities(*m_cPlatformHandle);
		if (_numElem > pC.SYS_LICENSED_CHANNELS*pC.MUX_RATIO)
		{
			CS_THROW("NumElements can not exceed " << pC.SYS_LICENSED_CHANNELS*pC.MUX_RATIO);
		}
		if (_endDepth > 0.3)
		{
			CS_THROW("EndDepth can not exceed 0.3");
		}
	}

	void UsIntCephasonicsBtcc::setupScan()
	{
		bool swModeEn = true;                      //need to set this to true when making fully-custom beams
		_scan = &ScanDef::createScanDef(*m_cPlatformHandle,
			*_probe,
			swModeEn,
			SP_SOUND,
			RX_BUFFER_SZ); //ignored in CHCAP mode

//controlled by the -C flag, default is 40MHz or can be 20MHz
		if (_clock == 20)
		{
			_scan->update(SetClock(RX, INP_CLK1, false, 2, 40000000));
			_scan->update(SetClock(TX, INP_CLK1, false, 1, 20000000));
		}
		else
		{
			_scan->update(SetClock(RX, INP_CLK1, false, 1, 40000000));
			_scan->update(SetClock(TX, INP_CLK1, false, 1, 40000000));
		}

		const BeamDef*                 rxBeam;
		const BeamDef*                 txBeam;
		vector<bool>*                  txMap;
		const FiringDef*               firing;
		const BeamEnsembleDef*         beamEnsemble;
		vector<const BeamEnsembleDef*> beamEnsembles;

		initBeams();  //Inside initBeams is where all the custom beams are defined

		for (int i = 0; i < NUM_CUSTOM_BEAMS; i++)
		{
			txBeam = txBeamArr[i];
			rxBeam = rxBeamArr[i];
			txMap = txPiezoArr[i];

			firing = &FiringDef::createFiringDef(*m_cPlatformHandle,
				*txBeam,
				vector<const BeamDef*>(1, rxBeam),
				*txMap);

			beamEnsemble =
				&BeamEnsembleDef::createBeamEnsembleDef(*m_cPlatformHandle,
					0.0,          //PRF=0.0 sets to maximum PRF
								  //override by -p option is handled
								  //by SetPRF command below
					1,            //has to be set to one for 2D-scans
					*firing);

			beamEnsembles.push_back(beamEnsemble);
		}

		// Create the SubFrameDef
		uint32       beams = NUM_CUSTOM_BEAMS;
		uint32       npb = 1;          //ignored if making full custom beams
		double       txFreq = 2500000;    //ignored if making full custom beams

		double       startDepth = _startDepth;//controlled by -S flag, default=0.0m
		double       endDepth = _endDepth;  //controlled by -e flag, default=0.01m
		double       focalDepth = 0.07;

		double       txFstop = 3.0;   //ignored if making full custom beams
		double       rxFstop = 2.0;   //ignored if making full custom beams
		double       angle = 0;     //ignored if making full custom beams
		double       centerAngle = 0;     //ignored if making full custom beams

		rxWindowType rxWindow = RECTANGULAR;  //ignored if making full custom beams

		uint16   numCycles = _numCycles;   //set by -N flag

		// create a new SubFrameDef and provide it with the vector of BeamEnsembleDefs
		_subframe =
			&SubFrameDef::createSubFrameDef(*m_cPlatformHandle,
				LINEAR,    //not used for fully custom beams beam creation
				beams,
				npb,
				txFreq,
				numCycles,
				startDepth, endDepth, focalDepth,
				txFstop, rxFstop, rxWindow, angle, centerAngle,
				beamEnsembles);

		_frameDef = &FrameDef::createFrameDef(*m_cPlatformHandle,
			CHCAP,
			*_subframe);

		_scan->update(AddFrame(*_frameDef));

		//VGA gain controlled by -v flag
		vector < TGC_Profile *> profiles;
		//Adding TGC_Profile i.e. Depth and gain to the profiles vector;
		profiles.push_back(new TGC_Profile(0.01, _vgaGain));
		profiles.push_back(new TGC_Profile(0.02, _vgaGain));
		profiles.push_back(new TGC_Profile(0.03, _vgaGain));
		profiles.push_back(new TGC_Profile(0.04, _vgaGain));
		profiles.push_back(new TGC_Profile(0.05, _vgaGain));
		profiles.push_back(new TGC_Profile(0.06, _vgaGain));
		profiles.push_back(new TGC_Profile(0.07, _vgaGain));
		profiles.push_back(new TGC_Profile(0.08, _vgaGain));
		profiles.push_back(new TGC_Profile(0.09, _vgaGain));
		profiles.push_back(new TGC_Profile(0.10, _vgaGain));
		profiles.push_back(new TGC_Profile(0.11, _vgaGain));
		profiles.push_back(new TGC_Profile(0.12, _vgaGain));

		_frameDef->update(SetTGC(profiles,
			0,
			0.0,
			0.0,
			0.0,
			50000.0,
			200000.0));

		//Input impedance controlled by -i flag
		_scan->update(SetLNA(18.5, _inpimp));

		//controlled by -a
		_scan->update(SetAAF(_aaf));

		//controlled by -b flag
		if (_firBypass)
		{
			_scan->update(SetDecimFiltBypass(true));
		}

		//controlled by -d flag
		_scan->update(SetDecimFiltDecimation(_dec));

		if (_dec != 1)
		{
			vector<vector<double> > myMap = coeffs_map.at(_dec);
			_subframe->update(SetDecimFiltCoeffs(UsIntCephasonicsBtcc::coeffs_map.at(_dec), 0));
		}
		else
		{
			if (_firBypass == false)
				CS_THROW("FIR filtering must be bypassed when decimation is 1!");
		}

		_frameDef->update(SetTransmitVoltage(min(m_voltage, 60.0)));
		_scan->update(SetTimeout(500, true));

		//controlled by -H and -c flags
		double fc = _hpf;
		bool bypass = _HPFBypass;
		_scan->update(SetHPF(fc, bypass));

		//controlled by -T flag
		bool dataGenEnable = _tpEn;
		if (dataGenEnable)
		{
			_scan->update(SetTPG(dataGenEnable));
			if (!_decPresent)
				_scan->update(SetDecimFiltDecimation(1));
			if (!_firBypassPresent)
				_scan->update(SetDecimFiltBypass(true));
			if (!_HPFBypassPresent)
				_scan->update(SetHPF(fc, true));
		}

		//controlled by -p flag
		_frameDef->update(SetPRF(_prf, 1));

		PlatformCapabilities pC = USPlatformMgr::getPlatformCapabilities(*m_cPlatformHandle);
		vector <bool> txChannelVect;
		vector <bool> rxChannelVect;
		vector <bool> muxElementVect;
		vector <bool> muxSwitchVect;
		//TODO add this later iff neccessary
		/*if (_txChannel != "")
		{
		  txChannelVect = createChannelVector(_txChannel,pC.SYS_LICENSED_CHANNELS);
		  //controlled by -t flag
		  _scan->update(SetTXChannel(txChannelVect));
		}

		if (_rxChannel != "")
		{
		  rxChannelVect = createChannelVector(_rxChannel,pC.SYS_LICENSED_CHANNELS);
		  //controlled by -r flag
		  _scan->update(SetRXChannel(rxChannelVect));
		}

		if (_muxElement != "")
		{
		  muxElementVect = createChannelVector(_muxElement,pC.SYS_LICENSED_CHANNELS*pC.MUX_RATIO);
		  //controlled by -m flag
		  _scan->update(SetMuxElement(muxElementVect));
		}

		if (_muxSwitch != "")
		{
		  muxSwitchVect = createChannelVector(_muxSwitch,pC.SYS_LICENSED_CHANNELS);
		  //controlled by -s flag
		  _scan->update(SetMuxSwitch(muxSwitchVect));
		}*/

		m_cUSEngine->setScanDef(*_scan);
	}

	/*vector<bool> UltrasoundInterfaceCephasonicsBTCC::createChannelVector(string value, uint32 elements)
	{
		vector<bool> vect(elements,true);

		vector<string> tempVect;
		string val(value);
		CSUtils::split(val,tempVect,",");
		size_t found;

		for(unsigned int i=0;i<tempVect.size();i++)
		{
		  found = tempVect[i].find(":");
		  if (found==string::npos)
		  {
			if (atoi(tempVect[i].c_str())-1 < 0 || atoi(tempVect[i].c_str())-1 > (int)elements-1)
			  CS_THROW("Specified vector out of range, range starts from 1 and ends at " << elements);
			vect[atoi(tempVect[i].c_str())-1] = false;
		  }
		  else
		  {
			vector<string>tempVect2;
			CSUtils::split(tempVect[i],tempVect2,":");
			int stVal = atoi(tempVect2[0].c_str());
			if (stVal-1 < 0)
			  CS_THROW("Specified vector out of range, range starts from 1");
			int endVal = atoi(tempVect2[1].c_str());
			if (endVal-1 < 0 || endVal-1 > (int)elements-1)
			  CS_THROW("Specified vector out of range, range starts from 1 and ends at " << elements);
			for(int j=stVal;j<=endVal;j++)
			  vect[j-1] = false;
		  }
		}
		return vect;
	}*/

	void
		UsIntCephasonicsBtcc::initBeams()
	{
		uint32 i;
		PlatformCapabilities pC = USPlatformMgr::getPlatformCapabilities(*m_cPlatformHandle);
		//NUM_EFF_ELEMENTS is equal to the total number of muxed channels available in the connected
		//hardware, unless this number is reduced by the -n flag
		//The transmitDelayVector is set up to have a monotonically increasing delay per incremental
		//muxed channel

		//The offset value is specified in 4x the system clock (sampling clock), so quarter sample
		//granularity specification is ok.
		//Technically, the delay offset value has a range from 0 to 2^20-1, however,
		//because the receive buffer is defined by RX_BUFFER_SIZE, normally 2048, this is the receive area
		//where the TX delay values would make sense to restrict itself within.  Otherwise RX beamforming
		//could not work properly.  This RX_BUFFER_SIZE limiter is in 40MHz clock samples, but since 
		//delays are specified in quarter sampling clocks this range is RX_BUFFER_SIZE*4.
		//In this example, a Transmit delay is applied to each beam of equal amount starting from 1/2 the RX buffer.
		//Then, a RX delay is applied to offset the TX delay, making the net delay only 1/2 the RX buffer.
		//Hence, resulting in a strait line display.
		vector<double> myTransmitDelayVector;
		double tx_delay_offset = 0;
		//double tx_delay_offset = (RX_BUFFER_SZ*4.0)/2; //half the buffer offset
		double tx_delay_incr = ((RX_BUFFER_SZ*4.0) - tx_delay_offset) / NUM_CUSTOM_BEAMS / 3; //incremental delay for each beam
		for (i = 0; i < NUM_CUSTOM_BEAMS * _simuFireNumElem; i++)
		{
			myTransmitDelayVector.push_back(tx_delay_offset + i*tx_delay_incr);
		}

		//The transmit wave vector specifies the shape of the pulse to transmit.  In this example, an
		//active pusle or passive pusle is specified and based on the beam, one element is chosen to do
		//transmit while all other elements are set to a zero wave pulse.
		vector<vector<PulseVal> > myTransmitWaveVector;
		vector<PulseVal> myWaveDef_passive;
		vector<PulseVal> myWaveDef_active;

		for (i = 0; i < 64; i++)
		{
			myWaveDef_passive.push_back(GND);
		}

		//Creating TX single pulse, 2.5MHz frequency
		//The wave pulse is constructed at 4x the system clock.
		//Assuming the system clock is 40MHz, then below is an even duty cycle
		//32 positive, 32 negative pulse waveform.  NOTE:  The leading, trailing, and middle ground
		//points are required in order to make a proper wave.
		//The Wave Frequency is given by TXFREQ = SysClock*4/Number of pos pulses + Number of neg pulses
		//2.5MHz Transmit Pulse Signal = 40*4/(32+32)
		//Calculation of num_pules to make frequency of transmit pulse is based on _txFreq, set by -f option
		double tx_divisor = 1.0;
		uint32 num_pulses = ((double)_clock*1e6*pC.TX_OVERSAMPLE / _txFreq) / 2;

		//In the case where the total number of pulses used exceeds 247, this means that the wave table would be stored
		//to the advanced if long pulse memory.  This long pulse memory can not be delayed.  However, in a case where a lower
		//frequency is desired, which requires more than 247 pulse train, we instead use 247 pulse train in conjunction with
		//the tx frequency divider to achieve the desired transmit frequency.
		if (num_pulses * 2 > MAX_WAVE_TABLE_SHORT_PULSE_LEN) //in this case we have to use tx divisor to make the pulse
		{
			double MAX_FREQ = _clock*1e6*pC.TX_OVERSAMPLE / (MAX_WAVE_TABLE_SHORT_PULSE_LEN);
			tx_divisor = _txFreq / MAX_FREQ;
			num_pulses = MAX_WAVE_TABLE_SHORT_PULSE_LEN / 2;
		}

		myWaveDef_active.push_back(GND); //Leading Ground
		for (i = 0; i < num_pulses; i++)
		{
			myWaveDef_active.push_back(POSV0);
		}
		myWaveDef_active.push_back(GND); //Mid Ground
		for (i = 0; i < num_pulses; i++)
		{
			myWaveDef_active.push_back(NEGV0);
		}
		myWaveDef_active.push_back(GND); //End Ground

		//NOTE: RX Delay vector is not used in CHCAP mode
		//The RX Delay vector is specified in 4x the system clock.
		//Offset the RX Delay to compensate for the TX delay applied.
		//Since TX delay is positive, RX delay is negative to compensate
		vector<vector<double> > myRxDelayVector;
		double rx_delay_incr = -1.0*tx_delay_incr;

		uint16 vector_entries = (NUM_CUSTOM_BEAMS > pC.MAX_RX_BEAMS_SWMODE / 2) ? 8 : 16;
		for (i = 0; i < NUM_CUSTOM_BEAMS * _simuFireNumElem; i++)
		{
			myRxDelayVector.push_back(vector<double>(vector_entries, i*rx_delay_incr));
		}

		//NOTE: RX Weight vector is not used in CHCAP mode
		vector<vector<double> > myRxWeightVector;
		vector<double> myRxWeightVector_active(vector_entries, 1.0);
		vector<double> myRxWeightVector_passive(vector_entries, 0.0);
		//myTxPiezo vector tells which piezo elements need to be turned on per beam
		//myTXPiezo has to be size of the number of elements per probe
		//myTXPiezo on elements can not exceed number of system channels available.
		vector<bool> myTxPiezoVector(NUM_EFF_ELEMENTS, false);
		for (uint32 beam_idx = 0; beam_idx < NUM_CUSTOM_BEAMS; beam_idx++)
		{

			//Selecting start and end element for python plot
			//NOTE:  CURRENT IMPLEMENTATION IS ONLY FOR PLATFORMS WITH EVEN CHANNEL COUNTS!
			uint16 start_elem = 0;
			uint16 end_elem = NUM_EFF_ELEMENTS;
			if (NUM_EFF_ELEMENTS > pC.SYS_LICENSED_CHANNELS)
			{
				start_elem = (beam_idx * 2 * _simuFireNumElem >= pC.SYS_LICENSED_CHANNELS) ?
					beam_idx*_simuFireNumElem - ((pC.SYS_LICENSED_CHANNELS / 2) - 1) : 0;
				end_elem = start_elem + pC.SYS_LICENSED_CHANNELS;
			}

			//_fullAperture controlled by -A flag
			for (uint32 el_idx = 0; el_idx < NUM_EFF_ELEMENTS; el_idx++)
			{
				//if (el_idx == beam_idx)
				if ((el_idx >= beam_idx * _simuFireNumElem) &&
					(el_idx < beam_idx * _simuFireNumElem + _simuFireNumElem))
				{
					if (!_fullAperture)
					{
						//myTxPiezoVector[el_idx] = true;
						myTransmitWaveVector.push_back(myWaveDef_active);
						myRxWeightVector.push_back(myRxWeightVector_active);
					}
				}
				else
				{
					if (!_fullAperture)
					{
						//myTxPiezoVector[el_idx] = false;
						myTransmitWaveVector.push_back(myWaveDef_passive);
						myRxWeightVector.push_back(myRxWeightVector_passive);
					}
				}
				if (_fullAperture)
				{
					myTransmitWaveVector.push_back(myWaveDef_active);
					myRxWeightVector.push_back(myRxWeightVector_active);
				}
				//if (_fullAperture)
				//{
				if (el_idx >= start_elem && el_idx < end_elem) myTxPiezoVector[el_idx] = true;
				//}
			}
			txBeamArr[beam_idx] = &BeamDef::createTXBeamDef(*m_cPlatformHandle, myTransmitWaveVector,
				myTransmitDelayVector, tx_divisor);
			rxBeamArr[beam_idx] = &BeamDef::createRXBeamDef(*m_cPlatformHandle, myRxDelayVector, myRxWeightVector);
			txPiezoArr[beam_idx] = new vector<bool>(myTxPiezoVector);
			myTransmitWaveVector.clear();
			myRxWeightVector.clear();
			for (uint32 j = 0; j < myTxPiezoVector.size(); j++) { myTxPiezoVector[j] = false; }
		}
	}

	UsIntCephasonicsBtcc::UsIntCephasonicsBtcc(tbb::flow::graph & graph, const std::string& nodeID)
		: AbstractInput<RecordObject>(graph, nodeID)
	{
		m_ready = false;

		if (!m_environSet)
		{
			setenv("CS_CCMODE", "1", true);
			setenv("CS_LOGFILE", "", true);
			m_environSet = true;
		}

		//Setup allowed values for parameters
		m_valueRangeDictionary.set<double>("voltage", 6, 60, 6, "Voltage");
		m_valueRangeDictionary.set<double>("intensityScaling", 0.0001, 10000.0, 20.0, "Intensity scaling");
		m_valueRangeDictionary.set<string>("layout", { "linear", "64x", "32x" }, "linear", "Layout");

		setProgramOptions();
	}

	UsIntCephasonicsBtcc::~UsIntCephasonicsBtcc()
	{
		//End of the world, waiting to join completed thread on teardown
		m_cUSEngine->tearDown();   //Teardown USEngine
		if (m_runEngineThread.joinable())
		{
			m_runEngineThread.join();
		}
	}

	void UsIntCephasonicsBtcc::initializeDevice()
	{
		m_voltage = m_configurationDictionary.get<double>("voltage");

		//Step 1 ------------------ "Setup Platform"
		setProgramOptions();
		m_cPlatformHandle = setupPlatform();

		checkOptions();

		m_cUSEngine = unique_ptr<USEngine>(new USEngine(*m_cPlatformHandle));
		m_cUSEngine->stop();
		m_cUSEngine->setBlocking(true);

		//Step 2 ----------------- "Create Scan Definition"
		setupProbe();
		setupScan();

		//Step 3 ----------------- "Create Ultrasound Engine Thread"
		//create the data processor that later handles the data
		m_pDataProcessor = unique_ptr<UsIntCephasonicsBtccProc>(
			new UsIntCephasonicsBtccProc(*m_cPlatformHandle, this)
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

	void UsIntCephasonicsBtcc::startAcquisition()
	{
		m_cUSEngine->start();
	}

	void UsIntCephasonicsBtcc::stopAcquisition()
	{
		m_cUSEngine->stop();       //Stop USEngine
	}

	void UsIntCephasonicsBtcc::configurationEntryChanged(const std::string & configKey)
	{
		lock_guard<mutex> lock(m_objectMutex);
		if (configKey == "voltage")
		{
			m_voltage = m_configurationDictionary.get<double>("voltage");
			if (m_ready)
			{
				_frameDef->update(SetTransmitVoltage(min(m_voltage, 60.0)));
			}
		}
		else if (configKey == "intensityScaling")
		{
			m_intensityScaling = m_configurationDictionary.get<double>("intensityScaling");
		}
		else if (configKey == "layout")
		{
			m_layout = m_configurationDictionary.get<string>("layout");

			if (m_layout == "64x" || m_layout == "32x")
			{
				size_t numChans;
				if (m_layout == "64x")
				{
					numChans = 64;
				}
				else if (m_layout == "32x")
				{
					numChans = 32;
				}
				size_t numRows = 1152 / numChans;


				m_pImageProperties = make_shared<USImageProperties>(
					vec2s{ numChans, 1 },
					numRows,
					USImageProperties::ImageType::BMode,
					USImageProperties::ImageState::RF,
					USImageProperties::TransducerType::Linear,
					numRows);
			}
		}
	}

	void UsIntCephasonicsBtcc::configurationChanged()
	{
		readConfiguration();
	}

	void UsIntCephasonicsBtcc::readConfiguration()
	{
		lock_guard<mutex> lock(m_objectMutex);
		//read conf values
		m_voltage = m_configurationDictionary.get<double>("voltage");
		m_intensityScaling = m_configurationDictionary.get<double>("intensityScaling");

		m_layout = m_configurationDictionary.get<string>("layout");
		if (m_layout == "64x" || m_layout == "32x")
		{
			size_t numChans;
			if (m_layout == "64x")
			{
				numChans = 64;
			}
			else if (m_layout == "32x")
			{
				numChans = 32;
			}
			size_t numRows = 1152 / numChans;


			m_pImageProperties = make_shared<USImageProperties>(
				vec2s{ numChans, 1 },
				numRows,
				USImageProperties::ImageType::BMode,
				USImageProperties::ImageState::RF,
				USImageProperties::TransducerType::Linear,
				numRows);
		}
		else
		{
			//TODO this is just to have somthing. Needs to be deduced / read from cephasonics
			//prepare the USImageProperties
			m_pImageProperties = make_shared<USImageProperties>(
				vec2s{ 128, 1 },
				512,
				USImageProperties::ImageType::BMode,
				USImageProperties::ImageState::RF,
				USImageProperties::TransducerType::Linear,
				60);

		}
	}

	void UsIntCephasonicsBtcc::layoutChanged(ImageLayout & layout)
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
				USImageProperties::ImageState::RF,
				USImageProperties::TransducerType::Linear,
				30);
		}
	}

	void UsIntCephasonicsBtcc::putData(uint16_t platformIndex, uint32_t numChannels, size_t numSamples, size_t numBeams, int16_t* data)
	{
		static CallFrequency m("US");
		double timestamp = getCurrentTime();

		static vector<bool> platformsReceived;
		static size_t sNumElems = 0;
		static size_t sNumSamples = 0;
		static vector<int16_t> buff;

		if (platformsReceived.size() != _numPlatforms)
		{
			platformsReceived.resize(_numPlatforms, false);
		}
		if ((numBeams*_simuFireNumElem) != sNumElems || numSamples != sNumSamples)
		{
			sNumElems = numBeams * _simuFireNumElem;
			sNumSamples = numSamples;
			buff.clear();
			buff.resize(sNumElems*sNumSamples, 0);

			platformsReceived.assign(_numPlatforms, false);
		}

		size_t numVectors = sNumElems;

		platformIndex = platformIndex % _numPlatforms;

		//build filename
		/*std::stringstream filename;
		filename << "/mnt/data/ascii_test/scandata_" << platformIndex << ".txt";
		writeAscii(filename.str(), data, numChannels*numSamples*numBeams);*/

		for (size_t beam = 0; beam < numBeams; beam++)
		{
			size_t firstElemContained = beam     *_simuFireNumElem;
			size_t lastElemContained = (beam + 1)*_simuFireNumElem - 1;
			//for(size_t firing = 0; firing < _simuFireNumElem; firing++)
			//{
			//	size_t elem = beam * _simuFireNumElem;
			for (size_t elem = firstElemContained; elem <= lastElemContained; elem++)
			{
				size_t elemPlatform = (elem / numChannels) % _numPlatforms;
				if (elemPlatform == platformIndex)
				{
					size_t channel = elem % numChannels;
					for (size_t y = 0; y < numSamples; y++)
					{
						int16_t rawVal = data[channel + y*numChannels + beam*numChannels*numSamples];

						buff[elem + y*sNumElems] = rawVal;
					}
				}
			}
		}
		platformsReceived[platformIndex] = true;

		bool allreceived = true;
		for (bool r : platformsReceived)
		{
			allreceived = allreceived && r;
		}
		if (allreceived)
		{
			lock_guard<mutex> lock(m_objectMutex);
			m.measure();

			if (m_layout == "linear")
			{
				shared_ptr<Container<uint8_t> > pData = make_shared<Container<uint8_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), numSamples * numVectors);
				std::memset(pData->get(), 0, sizeof(uint8_t)*numSamples*numVectors);
				for (size_t vect = 0; vect < numVectors; vect++)
				{
					for (size_t y = 0; y < numSamples; y++)
					{
						int16_t rawVal = buff[vect + y*numVectors];
						uint16_t absVal = m_intensityScaling*log10((double)(abs(rawVal)) + 1);
						uint8_t val = static_cast<uint8_t>(min(absVal, (uint16_t)255));

						pData->get()[vect + y*numVectors] = val;
					}
				}

				shared_ptr<USImage<uint8_t> > pImage;
				pImage = make_shared<USImage<uint8_t> >(
					vec2s{ numVectors, numSamples }, pData, m_pImageProperties, timestamp, timestamp);
				addData<0>(pImage);
			}
			else if (m_layout == "64x" || m_layout == "32x")
			{
				size_t numChans;
				if (m_layout == "64x")
				{
					numChans = 64;
				}
				else if (m_layout == "32x")
				{
					numChans = 32;
				}
				size_t numRows = numVectors / numChans;
				shared_ptr<Container<uint8_t> > pData = make_shared<Container<uint8_t> >(ContainerLocation::LocationHost, ContainerFactory::getNextStream(), numChans * numRows);
				std::memset(pData->get(), 0, sizeof(uint8_t)*numChans*numRows);
				for (size_t row = 0; row < numRows; row++)
				{
					for (size_t chan = 0; chan < numChans; chan++)
					{
						uint32_t sum = 0;
						for (size_t y = 0; y < numSamples; y++)
						{
							int16_t rawVal = buff[chan + numChans*row + y*numVectors];
							sum += abs(rawVal);
						}
						uint32_t absVal = m_intensityScaling*(double)(abs(sum)) / numSamples;
						uint8_t val = static_cast<uint8_t>(min(absVal, (uint32_t)255));

						pData->get()[chan + numChans*row] = val;
					}
				}

				shared_ptr<USImage<uint8_t> > pImage;
				pImage = make_shared<USImage<uint8_t> >(
					vec2s{ numChans, numRows }, pData, m_pImageProperties, timestamp, timestamp);
				addData<0>(pImage);
			}
			platformsReceived.assign(_numPlatforms, false);
			buff.assign(numVectors*numSamples, 0);

		}

	}

	bool UsIntCephasonicsBtcc::ready()
	{
		return m_ready;
	}

	void UsIntCephasonicsBtcc::freeze()
	{
		m_cUSEngine->stop();
	}

	void UsIntCephasonicsBtcc::unfreeze()
	{
		m_cUSEngine->start();
	}
}
