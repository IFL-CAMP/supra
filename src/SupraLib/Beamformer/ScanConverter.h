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

#ifndef __SCANCONVERTER_H__
#define __SCANCONVERTER_H__

#include "USImage.h"
#include "Container.h"

#include <memory>

namespace supra
{
	class ScanConverter
	{
	public:
		typedef uint32_t IndexType;
		typedef float WeightType;

		static constexpr int m_mappingMaxIterations = 1000;
		static constexpr double m_mappingDistanceThreshold = 1e-5;

		template<typename InputType, typename OutputType>
		std::shared_ptr<Container<OutputType> >
			convert(const std::shared_ptr<USImage> & inImage);
		std::shared_ptr<Container<uint8_t> > getMask();
		void updateInternals(const std::shared_ptr<const USImageProperties> & inImageProps);
		vec3s getImageSize() const { return m_imageSize; }
		bool is2D() const { return m_is2D; }

	private:
		static constexpr double m_skewnessTestThreshold = 1e-6;

		double barycentricCoordinate2D(const vec2& a, const vec2& b, const vec2& c);
		//double barycentricCoordinate3D(const vec& a, const vec& b, const vec& c, const vec& p);
		bool pointInsideTriangle(const vec2& a, const vec2& b, const vec2& c, const vec2& p);
		//bool pointInsideTetrahedron(const vec& a, const vec& b, const vec& c, const vec& d, const vec& p);

		vec pointLineConnection(const vec& a, const vec& da, const vec& x);
		//vec pointPlaneConnection(const vec& a, const vec& na, const vec& x);
		vec2 mapToParameters2D(const vec& a, const vec& b, const vec& da, const vec& db, double startDepth, double endDepth, const vec& x);
		/*std::pair<vec, bool>  mapToParameters3D(
			const vec& a, const vec& ax, const vec& ay, const vec& axy,
			const vec& da, const vec& dax, const vec& day, const vec& daxy,
			double startDepth, double endDepth, const vec& x);*/

		bool m_is2D;

		std::shared_ptr<Container<uint8_t> > m_mask;
		std::shared_ptr<Container<IndexType> > m_sampleIdx;
		std::shared_ptr<Container<WeightType> > m_weightX;
		std::shared_ptr<Container<WeightType> > m_weightY;
		std::shared_ptr<Container<WeightType> > m_weightZ;

		vec m_bbMin = { 0,0,0 };
		vec m_bbMax = { 0,0,0 };
		vec3s m_imageSize = { 0,0,0 };
	};
}

#endif //!__SCANCONVERTER_H__
