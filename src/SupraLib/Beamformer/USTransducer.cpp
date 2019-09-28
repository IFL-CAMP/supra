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

#include "USTransducer.h"

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <assert.h>

using namespace std;

namespace supra
{
	USTransducer::USTransducer(
		size_t numElements,
		vec2s elementLayout,
		USTransducer::Type type,
		const vector<double>& pitchX,
		const vector<double>& pitchY,
		const vector<std::pair<double, double> >& matchingLayers,
		const vector<bool>& elementMap,
		const vector<int32_t>& elementToChannelMap)
		: m_numElements(numElements)
		, m_elementLayout(elementLayout)
		, m_type(type)
		, m_pitchX(pitchX)
		, m_pitchY(pitchY)
		, m_matchingLayers(matchingLayers)
		, m_elementMap(elementMap)
		, m_elementToChannelMap(elementToChannelMap)
	{
		assert(m_elementMap.size() == 0 || m_elementMap.size() == m_numElements);
		assert(m_elementToChannelMap.size() == 0 || m_elementToChannelMap.size() == m_numElements);

		if (m_elementMap.size() == 0)
		{
			m_elementMap = vector<bool>(m_numElements, true);
		}
		computeInternals();
	}

	void USTransducer::setType(USTransducer::Type type)
	{
		m_type = type;
		computeInternals();
	}

	void USTransducer::setNumElements(size_t numElements)
	{
		m_numElements = numElements;
		computeInternals();
	}

	void USTransducer::setPitch(const vector<double>& pitchX, const vector<double>& pitchY)
	{
		m_pitchX = pitchX;
		m_pitchY = pitchY;
		computeInternals();
	}

	USTransducer::Type USTransducer::getType() const
	{
		return m_type;
	}

	size_t USTransducer::getNumElements() const
	{
		return m_numElements;
	}

	vec2s USTransducer::getElementLayout() const
	{
		return m_elementLayout;
	}

	shared_ptr<const vector<vec>> USTransducer::getElementCenterPoints() const
	{
		return static_cast<shared_ptr<const vector<vec> >>(m_elementCenterPoints);
	}

	shared_ptr<const vector<vec>> USTransducer::getElementNormals() const
	{
		return static_cast<shared_ptr<const vector<vec> >>(m_elementNormals);
	}

	const std::vector<bool>& USTransducer::getElementMap() const
	{
		return m_elementMap;
	}

	bool USTransducer::hasSpecificParameter(std::string parameterName) const
	{
		return m_specificParameters.find(parameterName) == m_specificParameters.end();
	}

	const std::string & USTransducer::getSpecificParameter(std::string parameterName) const
	{
		//TODO test how it behaves if we access something not in the map (should result in an exception)
		return m_specificParameters.find(parameterName)->second;
	}

	const std::map<std::string, std::string>& USTransducer::getSpecificParameters() const
	{
		return m_specificParameters;
	}

	bool USTransducer::is2D() const
	{
		return true;
	}

	bool USTransducer::isSparse() const
	{
		// an array is considered sparse, if not all elements in the layout are present.
		return std::any_of(m_elementMap.begin(), m_elementMap.end(), [](bool map) -> bool {return !map; });
	}

	std::vector<int32_t> USTransducer::getMarkedElementToChannelMap() const
	{
		std::vector<int32_t> markedMap(m_elementMap.size());
		std::transform(m_elementMap.begin(), m_elementMap.end(), m_elementToChannelMap.begin(), markedMap.begin(),
			[](const bool& map, const int32_t& index) -> int32_t { return map ? index : ElementChannelMapNotConnected; });
		return markedMap;
	}

	double USTransducer::computeTransitTime(vec2s elementIndex, vec elementToTarget, double speedOfSoundMMperS, bool correctForMatchingLayer) const
	{
		double transitTime = 0.0;
		if (m_matchingLayers.size() == 0 || !correctForMatchingLayer)
		{
			transitTime = norm(elementToTarget) / speedOfSoundMMperS;
		}
		else
		{
			// find deflection w.r.t. element normal
			auto normal = (*m_elementNormals)[elementIndex.x + m_elementLayout.x*elementIndex.y];
			vec3 normalFixed{normal.x, normal.z, normal.y};
			double p_axial = dot(elementToTarget, normalFixed);
			double p_deflection = std::sqrt(sq(norm(elementToTarget)) - sq(p_axial));
			vec2 targetPos{ p_deflection, p_axial };

			auto ray = findRay(targetPos, speedOfSoundMMperS);
			transitTime = computeRayTime(targetPos, ray.second, speedOfSoundMMperS);
		}
		return transitTime;
	}

	shared_ptr<const vector<vec4>> USTransducer::getElementCenterPointsHom() const
	{
		shared_ptr<vector<vec4> > startPointHom = make_shared<vector<vec4> >(m_elementCenterPoints->size());
		transform(m_elementCenterPoints->begin(), m_elementCenterPoints->end(), startPointHom->begin(),
			[](const vec& a) -> vec4 {
			return a.pointToHom();
		});

		return static_cast<shared_ptr<const vector<vec4> >>(startPointHom);
	}

	shared_ptr<const vector<vec4>> USTransducer::getElementNormalsHom() const
	{
		shared_ptr<vector<vec4> > normalsHom = make_shared<vector<vec4> >(m_elementNormals->size());
		transform(m_elementNormals->begin(), m_elementNormals->end(), normalsHom->begin(),
			[](const vec& a) -> vec4 {
			return a.vectorToHom();
		});

		return static_cast<shared_ptr<const vector<vec4> >>(normalsHom);
	}

	void USTransducer::computeInternals()
	{
		//This is where the main magic happens: Based on the main attributes, we compute
		//	m_elementCenterPoints and 
		//	m_elementNormals
		m_elementCenterPoints = make_shared<vector<vec> >();
		m_elementNormals = make_shared<vector<vec> >();
		m_elementCenterPoints->resize(m_numElements);
		m_elementNormals->resize(m_numElements);

		switch (m_type) {
		case Linear:
		{
			//Linear transducers are defined to have their elements on the x-Axis and scan lines go along positive y-Axis
			assert(m_pitchX.size() == m_numElements - 1);
			assert(m_pitchY.size() == 0);
			double widthX = accumulate(m_pitchX.begin(), m_pitchX.end(), 0.0);

			double currentPositionX = -widthX / 2;
			for (size_t i = 0; i < m_numElements; i++)
			{
				m_elementCenterPoints->at(i) = vec{ currentPositionX, 0, 0 };
				m_elementNormals->at(i) = vec{ 0, 1, 0 };

				if (i < m_numElements - 1)
				{
					currentPositionX += m_pitchX[i];
				}
			}
		}
		break;
		case Planar:
		{
			//planar transducers are defined to have rows of elements on the x-Axis which are stacked along the y-Axis
			assert(m_pitchX.size() == m_elementLayout.x - 1);
			assert(m_pitchY.size() == m_elementLayout.y - 1);
			vec2 width = {
					accumulate(m_pitchX.begin(), m_pitchX.end(), 0.0),
					accumulate(m_pitchY.begin(), m_pitchY.end(), 0.0)
			};

			vec2 currentPosition = -width / 2;
			for (size_t elemIdxY = 0; elemIdxY < m_elementLayout.y; elemIdxY++)
			{
				currentPosition.x = -(width.x) / 2;
				for (size_t elemIdxX = 0; elemIdxX < m_elementLayout.x; elemIdxX++)
				{
					m_elementCenterPoints->at(elemIdxX + m_elementLayout.x*elemIdxY) =
						vec{ currentPosition.x,
							 currentPosition.y, 0 };
					m_elementNormals->at(elemIdxX + m_elementLayout.x*elemIdxY) = vec{ 0, 1, 0 };

					if (elemIdxX < m_elementLayout.x - 1)
					{
						currentPosition.x += m_pitchX[elemIdxX];
					}
				}

				if (elemIdxY < m_elementLayout.y - 1)
				{
					currentPosition.y += m_pitchY[elemIdxY];
				}
			}
		}
		break;
		/*case Curved:
			//For curved arrays, the center element (or the middle between the two center elements in case of even scan line number)
			// is defined to be in the center of the image, at y = 0.
			// With that and the angle of the scanlines variing from -m_fov/2 to m_fov/2 we can compute the points and directions.
			for (size_t i = 0; i < m_numElements; i++)
			{
				double angle = pitch*(i - ((m_numElements - 1) / 2.0)) / m_probeRadius;
				m_elementCenterPoints->at(i) = vec{ m_probeRadius*sin(angle) , m_probeRadius*cos(angle)-m_probeRadius, 0 };
				m_elementNormals->at(i) = vec{ sin(angle), cos(angle), 0 };
			}
			break;*/
		default:
			throw new std::out_of_range("Transducer type not implemented yet. In USImageProperties::computeInternals");
		}
	}

	std::pair<double, std::vector<vec2> >
		USTransducer::findRay(vec2 pos, double speedOfSoundMMperS) const
	{
		double stoppingDistance = 1e-7;
		size_t maxIterations = 1000;

		double distance = std::numeric_limits<double>::max();
		double angle = degToRad(45.0);
		double limitLower = degToRad(0.0);
		double limitUpper = degToRad(89.0);
		std::pair<vec2, std::vector<vec2> > ray;
		for (size_t numIter = 0; numIter < maxIterations && (abs(distance) > stoppingDistance); numIter++)
		{
			ray = computeRay(angle, speedOfSoundMMperS, pos.y);
			distance = pos.x - ray.first.x;

			if (abs(distance) > stoppingDistance)
			{
				if (distance < 0) // Angle was too large
				{
					limitUpper = angle;
				}
				else { // angle was too small
					limitLower = angle;
				}
				angle = (limitLower + limitUpper) / 2;
			}
		}

		return make_pair(angle, ray.second);
	}
	
	std::pair<vec2, std::vector<vec2> >
		USTransducer::computeRay(double alpha, double speedOfSoundMMperS, double posAxial) const
	{
		size_t n = m_matchingLayers.size();

		std::vector<vec2> rayPositions;
		//add the first matching layer
		rayPositions.push_back({ tan(alpha) * m_matchingLayers[0].first, m_matchingLayers[0].first });

		// find the positions of the intersections with the remaining matching layers
		double lastAlpha = alpha;
		for (size_t k = 1; k < n; k++)
		{
			auto layer = matchingLayer(rayPositions[k -1], lastAlpha, m_matchingLayers[k-1].second, m_matchingLayers[k].second, m_matchingLayers[k].first);
			auto q = layer.first;
			auto angle2 = layer.second;
			rayPositions.push_back(q);
			lastAlpha = angle2;
		}

		auto tissueLayer = matchingLayer(rayPositions[n-1], lastAlpha, m_matchingLayers[n-1].second, speedOfSoundMMperS, posAxial - rayPositions[n-1].y);
		return make_pair(tissueLayer.first, rayPositions);
	}

	double USTransducer::computeRayTime(const vec2& endPoint, const std::vector<vec2> & rayPositions, double speedofSoundMMperS) const
	{
		double time = 0.0;
		vec2 lastPoint{ 0.0, 0.0 };
		for (size_t k = 0; k < rayPositions.size(); k++)
		{
			time += norm(rayPositions[k] - lastPoint) / m_matchingLayers[k].second;
			lastPoint = rayPositions[k];
		}
		time += norm(endPoint - lastPoint) / speedofSoundMMperS;
		return time;
	}

	std::pair<vec2, double>
		USTransducer::matchingLayer(vec2 p, double angle1, double c1, double c2, double thickness) const
	{
		double angle2 = asin(sin(angle1) / c1 * c2);
		vec2 q = p + vec2{ tan(angle2)*thickness, thickness };
		return make_pair(q, angle2);
	}
}
