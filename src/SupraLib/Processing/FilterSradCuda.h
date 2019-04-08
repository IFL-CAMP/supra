// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Walter Simson 
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __FILTERSRADCUDA_H__
#define __FILTERSRADCUDA_H__

#include "USImage.h"

#include <memory>

namespace supra
{
	class FilterSradCuda
	{
	public:
		typedef float WorkType;

		template<typename InputType, typename OutputType>
		static std::shared_ptr<Container<OutputType> >
			process(const std::shared_ptr<const Container<InputType>>& imageData, 
				vec3s size, double eps, uint32_t numberIterations, 
				double lambda, double speckleScale, double speckleScaleDecay);
	};
}

#endif //!__FILTERSRADCUDA_H__
