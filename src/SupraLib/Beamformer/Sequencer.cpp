// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Christoph Hennersperger
//		Email c.hennersperger@tum.de
// 		and
// 		Rüdiger Göbl
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
	using std::shared_ptr;
	using std::make_shared;
	using std::pair;

	using namespace logging;

	Sequencer::Sequencer(size_t numBeamformers)
		: m_pTransducer(nullptr)
	{
		m_numBeamformers = numBeamformers;

		for (auto k=0; k<numBeamformers; ++k)
		{
			m_beamformers.push_back(make_shared<Beamformer>());
			m_imageProperties.push_back(make_shared<USImageProperties>());
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
			it->setTransducer(transducer);
		}
	}

	void Sequencer::setUSImgProperties(const size_t bfUID, std::shared_ptr<USImageProperties> usImgProperties)
	{
		if (bfUID < m_numBeamformers)
		{
			m_imageProperties.at(bfUID) = usImgProperties;
		}
		else
		{
			logging::log_error("Error: Beamformer with wrong UID selected in Sequencer.");
		}
	}
	

	// get a specific beamformer within sequencer
	std::shared_ptr<Beamformer> Sequencer::getBeamformer(const size_t bfUID)
	{
		if (bfUID < m_numBeamformers)
		{
			return m_beamformers.at(bfUID);
		}
		else
		{
			logging::log_error("Error: Beamformer with wrong UID selected in Sequencer.");
			return make_shared<Beamformer>();
		}
	}


	std::shared_ptr<USImageProperties> Sequencer::getUSImgProperties(const size_t bfUID)
	{
		if (bfUID < m_numBeamformers)
		{
			return m_imageProperties.at(bfUID);
		}
		else
		{
			logging::log_error("Error: beamformer with wrong UID selected in Sequencer.");
			return std::shared_ptr<USImageProperties>(nullptr);
		}
	}
}
