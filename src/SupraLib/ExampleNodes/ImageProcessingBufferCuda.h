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

#ifndef __IMAGEPROCESSINGBUFFERCUDA_H__
#define __IMAGEPROCESSINGBUFFERCUDA_H__

#include "USImage.h"

#include <memory>

namespace supra
{
	class ImageProcessingBufferCuda
	{
	public:
		typedef float WorkType;

		template<typename InputType, typename OutputType>
		static std::shared_ptr<Container<OutputType> >
			process(const std::shared_ptr<const Container<InputType> > & imageData, vec3s size, WorkType factor);
	};
}

#endif //!__IMAGEPROCESSINGBUFFERCUDA_H__
