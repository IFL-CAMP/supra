// ================================================================================================
// 
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License, version 2.1, as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this program.  If not, see
// <http://www.gnu.org/licenses/>.
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
