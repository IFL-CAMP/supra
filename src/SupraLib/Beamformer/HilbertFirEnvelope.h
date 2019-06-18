// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __HILBERTFIRENVELOPE_H__
#define __HILBERTFIRENVELOPE_H__

#include "Container.h"

#include <memory>

namespace supra
{
	class HilbertFirEnvelope
	{
	public:
		typedef float WorkType;

		HilbertFirEnvelope(size_t filterLength);
		~HilbertFirEnvelope();

		template<typename InputType, typename OutputType>
		std::shared_ptr<Container<OutputType> >
			demodulate(
				const std::shared_ptr<const Container<InputType> >& inImageData,
				int numScanlines, int numSamples);

	private:
		void prepareFilter();

		size_t m_filterLength;

		std::shared_ptr<Container<WorkType> > m_hilbertFilter;
	};
}

#endif //!__HILBERTFIRENVELOPE_H__
