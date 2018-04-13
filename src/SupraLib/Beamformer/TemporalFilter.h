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

#ifndef __TEMPORALFILTER_H__
#define __TEMPORALFILTER_H__

#include "USImage.h"

#include <memory>
#include <queue>

namespace supra
{
	class TemporalFilter
	{
	public:
		typedef float WorkType;

		template<typename InputType, typename OutputType>
		std::shared_ptr<Container<OutputType> >
			filter(
				const std::queue<std::shared_ptr<const ContainerBase> > & inImageData,
				vec3s size,
				const std::vector<double> weights);
	};
}

#endif //!__TEMPORALFILTER_H__
