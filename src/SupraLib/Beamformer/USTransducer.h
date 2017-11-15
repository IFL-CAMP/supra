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

#ifndef __USTRANSDUCER_H__
#define __USTRANSDUCER_H__

#include <memory>
#include <vector>
#include <map>
#include "vec.h"

#include <utilities/utility.h>

namespace supra
{
	class USTransducer
	{
	public:
		enum Type {
			Linear,
			Curved,
			Planar,
			Bicurved
		};

		USTransducer(
			size_t numElements,
			vec2s elementLayout,
			USTransducer::Type type,
			const std::vector<double>& pitchX,
			const std::vector<double>& pitchY,
			const std::vector<std::pair<double, double> >& matchingLayers = {});

		/// create the transducer object from an xml description
		//USTransducer(std::string xmlFilename);

	public:
		/////////////////////////////////////////////////////////////////////
		// simple setters for defining properties
		/////////////////////////////////////////////////////////////////////
		void setType(USTransducer::Type type);		// Defines the type of transducer
		void setNumElements(size_t numElements);		// number of transducer elements
		void setPitch(					// pitch of the transducer elemens. Distance of the transducer element centers for linear and curved arrays
			const std::vector<double>& pitchX,
			const std::vector<double>& pitchY);

		template <typename valueType>
		void setSpecificParameter(std::string parameterName, valueType value);	// set one interface-specific parameter

		/////////////////////////////////////////////////////////////////////
		// simple getters
		/////////////////////////////////////////////////////////////////////
		USTransducer::Type getType() const;		// Defines the type of transducer
		size_t getNumElements() const;			// number of transducer elements
		vec2s getElementLayout() const;			// the logical arrangement of the elements. E.g. for 2D: 32x32
		std::shared_ptr < const std::vector<vec> > getElementCenterPoints() const; // vector of points that mark the center of each transducer element
		std::shared_ptr < const std::vector<vec> > getElementNormals() const;      // vector of unit-vectors that describe the element normals

		bool  hasSpecificParameter(std::string parameterName) const;					// whether one interface-specific parameter exists
		const std::string&  getSpecificParameter(std::string parameterName) const;	// get one interface-specific parameter
		const std::map<std::string, std::string>&  getSpecificParameters() const;	// map to the interface-specific parameters

		/////////////////////////////////////////////////////////////////////
		// Dependent properties, i.e. they only have a getter that computes the return value
		/////////////////////////////////////////////////////////////////////
		bool is2D() const;

		double computeTransitTime(vec2s elementIndex, vec elementToTarget, double speedOfSoundMMperS, bool correctForMatchingLayer = true) const;

		std::shared_ptr < const std::vector<vec4> >  getElementCenterPointsHom() const;	// array of points in HOMOGENEOUS COORDINATES that mark the center of each transducer element. EXPENSIVE
		std::shared_ptr < const std::vector<vec4> >  getElementNormalsHom() const;	    // array of vectors in HOMOGENEOUS COORDINATES that describe the element normals. EXPENSIVE

	private:
		/////////////////////////////////////////////////////////////////////
		// Defining properties
		/////////////////////////////////////////////////////////////////////
		USTransducer::Type m_type;	// Defines the type of transducer
		size_t m_numElements;		// number of transducer elements
		vec2s m_elementLayout;      // the logical arrangement of the elements. E.g. for 2D: 32x32
		std::vector<double> m_pitchX; // pitch of the transducer elemens. Distance of the transducer element centers for linear and curved arrays
		std::vector<double> m_pitchY; // pitch of the transducer elemens. Distance of the transducer element centers for linear and curved arrays
		std::vector<std::pair<double, double> > m_matchingLayers; // List of the matching layers, as pairs of thickness [mm] and speed of sound [m/s]

		/////////////////////////////////////////////////////////////////////
		// Properties for efficient ray-Based operations
		/////////////////////////////////////////////////////////////////////
		std::shared_ptr < std::vector<vec> > m_elementCenterPoints; // vector of points that mark the center of each transducer element
		std::shared_ptr < std::vector<vec> > m_elementNormals;		// vector of unit-vectors that describe the element normals

		// Map for interface specific parameters, they do not define the image itself but its meaning
		std::map<std::string, std::string> m_specificParameters;

		void computeInternals(void);

		std::pair<double, std::vector<vec2> >
			findRay(vec2 pos, double speedOfSoundMMperS) const;
		std::pair<vec2, std::vector<vec2> >
			computeRay(double alpha, double speedOfSoundMMperS, double posAxial) const;
		double computeRayTime(const vec2& endPoint, const std::vector<vec2> & rayPositions, double speedofSoundMMperS) const;
		std::pair<vec2, double>
			matchingLayer(vec2 p, double angle1, double c1, double c2, double thickness) const;
	};

	template<typename valueType>
	inline void USTransducer::setSpecificParameter(std::string parameterName, valueType value)
	{
		m_specificParameters[parameterName] = stringify(value);
	}

	template<>
	inline void USTransducer::setSpecificParameter<std::string>(std::string parameterName, std::string value)
	{
		m_specificParameters[parameterName] = value;
	}
}

#endif //!__USTRANSDUCER_H__
