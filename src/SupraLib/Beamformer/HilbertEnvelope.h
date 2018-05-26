// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2018, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __IQDEMODULATOR_H__
#define __IQDEMODULATOR_H__

#include "Container.h"

#include <memory>
#include <cufft.h>

namespace supra
{
	class HilbertEnvelope
	{
	public:
		typedef float WorkType;

		HilbertEnvelope();
		~HilbertEnvelope();

		template<typename InputType, typename OutputType>
		std::shared_ptr<Container<OutputType> >
			computeHilbertEnvelope(
				const std::shared_ptr<const Container<InputType> >& inImageData,
				int numScanlines, int numSamples, uint32_t decimation);
		int decimatedSignalLength(int numSamples, uint32_t decimation);

	private:
		cufftHandle m_cufftHandleR2C;
		cufftHandle m_cufftHandleC2C;
		bool m_fftHavePlan;
		int m_fftPlanLength;
		int m_fftPlanBatch;
	};
}

#endif //!__IQDEMODULATOR_H__
