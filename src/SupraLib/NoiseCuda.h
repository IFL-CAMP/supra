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

#ifndef __NOISECUDA_H__
#define __NOISECUDA_H__

#include "USImage.h"

#include <memory>

namespace supra
{
	class NoiseCuda
	{
	public:
		typedef float WorkType;

		template<typename InputType, typename OutputType>
		static std::shared_ptr<Container<OutputType> >
			process(const std::shared_ptr<const Container<InputType> > & imageData, vec3s size,
				WorkType additiveUniformMin, WorkType additiveUniformMax,
				WorkType additiveGaussMean, WorkType additiveGaussStd,
				WorkType multiplicativeUniformMin, WorkType multiplicativeUniformMax,
				WorkType multiplicativeGaussMean, WorkType multiplicativeGaussStd,
				bool additiveUniformCorrelated, bool additiveGaussCorrelated,
				bool multiplicativeUniformCorrelated, bool multiplicativeGaussCorrelated);
	private:
		static std::shared_ptr<Container<WorkType> > makeNoiseCorrelated(const std::shared_ptr<const Container<WorkType>>& in,
			size_t width, size_t height, size_t depth);
	};
}

#endif //!__NOISECUDA_H__
