// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//		Rüdiger Göbl
//		Email r.goebl@tum.de
// and
//      Christoph Hennerpserger
// 		Email c.hennersperger@tum.de
//
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __SEQUENCER_H__
#define __SEQUENCER_H__

#include "Beamformer.h"

#include <memory>
#include <vector>

namespace supra
{
	class Sequencer
	{
	public:

		Sequencer(size_t numBeamformers);
		~Sequencer();

		std::shared_ptr<USImageProperties> getUSImgProperties(const size_t bfUID);
		std::shared_ptr<Beamformer> getBeamformer(const size_t bfUID);

		void setUSImgProperties(const size_t bfUID, std::shared_ptr<USImageProperties> usImgProperties);
		void setTransducer(const USTransducer* transducer);

	private:

		std::vector<std::shared_ptr<Beamformer>> m_beamformers; // <sequenId>
		std::vector<std::shared_ptr<USImageProperties>> m_imageProperties;
		size_t m_numBeamformers;

		const USTransducer* m_pTransducer;
	};
}

#endif //!__BEAMFORMER_H__
