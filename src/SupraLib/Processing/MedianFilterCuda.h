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

#ifndef __MEDIANFILTERCUDA_H__
#define __MEDIANFILTERCUDA_H__

#include "USImage.h"

#include <memory>

namespace supra
{
	class MedianFilterCuda
	{
	public:
		typedef float WorkType;

		template<typename InputType, typename OutputType>
		static std::shared_ptr<Container<OutputType> >
			process(
				const std::shared_ptr<const Container<InputType> > & imageData, const vec3s& size, const vec3s& filterSize);
	};
}

#endif //!__MEDIANFILTERCUDA_H__
