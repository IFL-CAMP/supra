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

#ifndef __TIMEGAINCOMPENSATION_H__
#define __TIMEGAINCOMPENSATION_H__

#include "USImage.h"

#include <memory>

namespace supra
{
	class TimeGainCompensation
	{
	public:
		typedef float WorkType;

		TimeGainCompensation()
			: m_curveSampled(nullptr)
		{

		};

		void setCurve(const std::vector<std::pair<double, double> >& curvePoints);

		template<typename InputType, typename OutputType>
		std::shared_ptr<Container<OutputType> >
			process(const std::shared_ptr<const Container<InputType> > & imageData, vec3s size, size_t workDimension);

	private:
		void sampleCurve(size_t numSamples);

		std::vector<std::pair<double, double> > m_curvePoints;
		std::unique_ptr<Container<WorkType> > m_curveSampled;
	};
}

#endif //!__TIMEGAINCOMPENSATION_H__
