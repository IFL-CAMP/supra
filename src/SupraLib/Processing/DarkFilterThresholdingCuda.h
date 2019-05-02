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

#ifndef __DARKFILTERTHRESHOLDINGCUDA_H__
#define __DARKFILTERTHRESHOLDINGCUDA_H__

#include "USImage.h"

#include <memory>

namespace supra
{
	class DarkFilterThresholdingCuda
	{
	public:
		typedef float WorkType;

		template<typename InputType, typename OutputType>
		static std::shared_ptr<Container<OutputType> >
			process(
				const std::shared_ptr<const Container<InputType> > & imageData, 
				vec3s size, double threshold);
	};
}

#endif //!__DARKFILTERTHRESHOLDINGCUDA_H__
