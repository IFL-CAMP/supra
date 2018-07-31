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

#ifndef __LOGCOMPRESSOR_H__
#define __LOGCOMPRESSOR_H__

#include "USImage.h"

#include <memory>

namespace supra
{
	class LogCompressor
	{
	public:
		typedef float WorkType;

		template<typename InputType, typename OutputType>
		std::shared_ptr<Container<OutputType> >
			compress(const std::shared_ptr<const Container<InputType> > & inImageData, vec3s size, double dynamicRange, double scale, double inMax);
	};
}

#endif //!__LOGCOMPRESSOR_H__
