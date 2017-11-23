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

#include "Sequencer.h"

#include <memory>
#include <utilities/Logging.h>

namespace supra
{
	using std::max;
	using std::round;
	using std::vector;
	using std::make_shared;
	using std::pair;

	using namespace logging;

	Sequencer::Sequencer(size_t numBeamformers)
		: m_pTransducer(nullptr)
	{
		m_numBeamformers = numBeamformers;

		for (auto k=0; k<numBeamformers; ++k)
		{
			m_beamformers.push_back(std::vector<shared_ptr<Beamformer>>(1,shared_ptr<Beamformer>(new Beamformer)));
			m_imageProperties.push_back(std::vector<shared_ptr<USImageProperties>>(1,shared_ptr<USImageProperties>(new USImageProperties)));
			
			EnsembleTxSteeringParameters steerProps;
			steerProps.numAngles = {1,1};
			steerProps.startAngle = {0,0};
			steerProps.endAngle = {0,0};	
			m_steeringProperties.push_back(steerProps);
		}	
	}

	Sequencer::~Sequencer()
	{
	}


	void Sequencer::setTransducer(const USTransducer* transducer)
	{
		// internal transducer (default)
		m_pTransducer = transducer;

		// transducer needs to be unique, update all beamformers at once
		for (auto it : m_beamformers)
		{
			for (auto sit : it)
			{
				sit->setTransducer(transducer);
			}
		}
	}

	void Sequencer::setUSImgProperties(const size_t bfUID, const size_t angleID, std::shared_ptr<USImageProperties> usImgProperties)
	{
		if (bfUID < m_numBeamformers)
		{
			m_imageProperties.at(bfUID).at(angleID) = usImgProperties;
		}
		else
		{
			logging::log_error("Error: Beamformer with wrong UID selected in Sequencer.");
		}
	}


	void Sequencer::setNumAngles(const size_t bfUID, const vec2s numAngles)
	{
		if (numAngles != m_steeringProperties[bfUID].numAngles || m_steeringProperties[bfUID].beamAngles.size()==0)
		{
			m_steeringProperties[bfUID].numAngles = numAngles;
			computeSteeringAngles(bfUID);
		}
	}


	void Sequencer::setStartAngle(const size_t bfUID, const vec2 startAngle)
	{
		if (startAngle != m_steeringProperties[bfUID].startAngle)
		{
			m_steeringProperties[bfUID].startAngle = startAngle;
			computeSteeringAngles(bfUID);
		}
	}

	void Sequencer::setEndAngle(const size_t bfUID, const vec2 endAngle)
	{
		if (endAngle != m_steeringProperties[bfUID].endAngle)
		{
			m_steeringProperties[bfUID].endAngle = endAngle;

			computeSteeringAngles(bfUID);
		}
	}


	void Sequencer::computeSteeringAngles(const size_t bfUID)
	{
		m_steeringProperties[bfUID].beamAngles.clear();

		// use first bf for angle seqeucne as template and erase the rest
		std::shared_ptr<Beamformer> bfTemplate = m_beamformers[bfUID][0];
		std::shared_ptr<USImageProperties> proTemplate = m_imageProperties[bfUID][0];
		m_beamformers[bfUID].clear();
		m_imageProperties[bfUID].clear();

		double angleIncrementX = (m_steeringProperties[bfUID].endAngle.x-m_steeringProperties[bfUID].startAngle.x) / (m_steeringProperties[bfUID].numAngles.x-1);
		double angleIncrementY = (m_steeringProperties[bfUID].endAngle.y-m_steeringProperties[bfUID].startAngle.y) / (m_steeringProperties[bfUID].numAngles.y-1);

		double angleY = m_steeringProperties[bfUID].startAngle.y;
		for (size_t iY = 0; iY < m_steeringProperties[bfUID].numAngles.y; iY++)
		{
			double angleX = m_steeringProperties[bfUID].startAngle.x;
			for (size_t iX = 0; iX < m_steeringProperties[bfUID].numAngles.x; iX++)
			{
				m_steeringProperties[bfUID].beamAngles.push_back({angleX, angleY});

				std::shared_ptr<Beamformer> bf(new Beamformer(bfTemplate));
				bf->setTxSteeringAngle({angleX, angleY});
				m_beamformers[bfUID].push_back(bf);

				std::shared_ptr<USImageProperties> props(new USImageProperties(*proTemplate));
				// todo: set steering properties in US image
				m_imageProperties[bfUID].push_back(props);

				angleX += angleIncrementX;
			}
			angleY += angleIncrementY;
		}
	}

	// get a specific beamformer within sequencer
	std::shared_ptr<Beamformer> Sequencer::getBeamformer(const size_t bfUID, const size_t angleID)
	{
		if (bfUID < m_numBeamformers)
		{
			return m_beamformers.at(bfUID).at(angleID);
		}
		else
		{
			logging::log_error("Error: Beamformer with wrong UID selected in Sequencer.");
			return make_shared<Beamformer>();
		}
	}


	/* void Sequencer::updateBeamformer(
		size_t bfUID,
		std::string scanType,
		vec2s numScanlines,
		vec2s rxScanlineSubdivision,
		vec2s maxActiveElements,
		vec2s maxTxElements,
		double depth,
		bool txFocusActive,
		double txFocusDepth,
		double txFocusWidth,
		double rxFocusDepth,
		double speedOfSound,
		double prf,
		vec2 fov) {

		Beamformer::ScanType selectedScantype;
		if (scanType == "linear") {
			selectedScantype = Beamformer::Linear;
		} else if (scanType == "phased") {
			selectedScantype = Beamformer::Phased;
		} else if (scanType == "biphased") {
			selectedScantype = Beamformer::Biphased;
		}

		if (bfUID < m_numBeamformers)
		{
			// update settings within beamformer.
			m_beamformers.at(bfUID)->setTransducer(m_pTransducer);
			m_beamformers.at(bfUID)->setParameters(selectedScantype, numScanlines, rxScanlineSubdivision,
								maxActiveElements, maxTxElements, depth, txFocusActive, txFocusDepth,
								txFocusWidth, txFocusDepth, speedOfSound, prf, fov);
			m_beamformers.at(bfUID)->computeTxParameters();
		}
		else
		{
			logging::log_error("Error: Beamformer with wrong UID selected in Sequencer.");
		}
	} */

	std::shared_ptr<USImageProperties> Sequencer::getUSImgProperties(const size_t bfUID, const size_t angleID)
	{
		if (bfUID < m_numBeamformers)
		{
			return m_imageProperties.at(bfUID).at(angleID);
		}
		else
		{
			logging::log_error("Error: beamformer with wrong UID selected in Sequencer.");
			return std::shared_ptr<USImageProperties>(nullptr);
		}
	}



	vec2s Sequencer::getNumAngles(const size_t bfUID) const
	{
		return m_steeringProperties[bfUID].numAngles;
	}

	vec2 Sequencer::getStartAngle(const size_t bfUID) const
	{
		return m_steeringProperties[bfUID].startAngle;
	}

	vec2 Sequencer::getEndAngle(const size_t bfUID) const
	{
		return m_steeringProperties[bfUID].endAngle;
	}
}
