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
	struct EnsembleTxSteeringParameters
	{
		vec2s numAngles;				// number of steered beams to be used for ensemble
		vec2 startAngle;				// start angle definition (angles equally spaced) in degree
		vec2 endAngle;					// end angle definition (angles equally spaed) in degree
		std::vector<vec2d> beamAngles;	// beam steering angles calculated from parameters above
	};

	class Sequencer
	{
	public:

		Sequencer(size_t numBeamformers);
		~Sequencer();

		std::shared_ptr<USImageProperties> getUSImgProperties(const size_t bfUID, const size_t angleID);
		std::shared_ptr<Beamformer> getBeamformer(const size_t bfUID, const size_t angleID);
		vec2s getNumAngles(const size_t bfUID) const;
		vec2 getStartAngle(const size_t bfUID) const;
		vec2 getEndAngle(const size_t bfUID) const;

		void setUSImgProperties(const size_t bfUID, const size_t angleID, std::shared_ptr<USImageProperties> usImgProperties);
		void setTransducer(const USTransducer* transducer);
		void setNumAngles(const size_t bfUID, const vec2s numAngles);
		void setStartAngle(const size_t bfUID, const vec2 startAngle);
		void setEndAngle(const size_t bfUID, const vec2 endAngle);

		/* void updateBeamformer(
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
				vec2 fov);

 */
	protected: 
		void computeSteeringAngles(const size_t bfUID);

	private:

		std::vector<std::vector<std::shared_ptr<Beamformer>>> m_beamformers; // <sequenId<angleId>>
		std::vector<std::vector<std::shared_ptr<USImageProperties>>> m_imageProperties;
		std::vector<EnsembleTxSteeringParameters> 		m_steeringProperties;
		size_t m_numBeamformers;

		const USTransducer* m_pTransducer;
	};
}

#endif //!__BEAMFORMER_H__
