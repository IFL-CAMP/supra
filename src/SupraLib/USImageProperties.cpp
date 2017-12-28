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

#include "USImageProperties.h"

#include <algorithm>
#include <stdexcept>

using namespace std;

namespace supra
{
	std::ostream& operator<< (std::ostream& os, const ScanlineRxParameters3D& params)
	{
		os << std::setprecision(9) << params.position.x << " "
			<< std::setprecision(9) << params.position.y << " "
			<< std::setprecision(9) << params.position.z << " "
			<< std::setprecision(9) << params.direction.x << " "
			<< std::setprecision(9) << params.direction.y << " "
			<< std::setprecision(9) << params.direction.z << " "
			<< params.maxElementDistance.x << " "
			<< params.maxElementDistance.y << " ";
		for (size_t k = 0; k < std::extent<decltype(params.txWeights)>::value; k++)
		{
			os << params.txParameters[k].firstActiveElementIndex.x << " "
				<< params.txParameters[k].firstActiveElementIndex.y << " "
				<< params.txParameters[k].lastActiveElementIndex.x << " "
				<< params.txParameters[k].lastActiveElementIndex.y << " "
				<< params.txParameters[k].txScanlineIdx << " "
				<< std::setprecision(9) << params.txParameters[k].initialDelay << " "
				<< std::setprecision(9) << params.txWeights[k] << " ";
		}
		return os;
	}

	std::istream& operator >> (std::istream& is, ScanlineRxParameters3D& params)
	{
		is >> params.position.x
			>> params.position.y
			>> params.position.z
			>> params.direction.x
			>> params.direction.y
			>> params.direction.z
			>> params.maxElementDistance.x
			>> params.maxElementDistance.y;
		for (size_t k = 0; k < std::extent<decltype(params.txWeights)>::value; k++)
		{
			is >> params.txParameters[k].firstActiveElementIndex.x
				>> params.txParameters[k].firstActiveElementIndex.y
				>> params.txParameters[k].lastActiveElementIndex.x
				>> params.txParameters[k].lastActiveElementIndex.y
				>> params.txParameters[k].txScanlineIdx
				>> params.txParameters[k].initialDelay
				>> params.txWeights[k];
		}
		return is;
	}


	USImageProperties::USImageProperties(
		vec2s scanlineLayout,
		size_t numSamples,
		USImageProperties::ImageType imageType,
		USImageProperties::ImageState imageState,
		USImageProperties::TransducerType transducerType,
		double depth)
		: m_numScanlines(scanlineLayout.x * scanlineLayout.y)
		, m_scanlineLayout(scanlineLayout)
		, m_numSamples(numSamples)
		, m_imageType(imageType)
		, m_imageState(imageState)
		, m_transducerType(transducerType)
		, m_depth(depth)
		, m_imageResolutionSet(false)
		, m_imageResolution(0)
		, m_numChannels(1)
	{
	}


		USImageProperties::USImageProperties()
		: m_numScanlines(0)
		, m_scanlineLayout(vec2s{ 0, 0 })
		, m_numSamples(0)
		, m_imageType(USImageProperties::ImageType::BMode)
		, m_imageState(USImageProperties::ImageState::Raw)
		, m_transducerType(USImageProperties::TransducerType::Linear)
		, m_depth(0.0)
		, m_imageResolutionSet(false)
		, m_imageResolution(0.0)
		, m_numChannels(1)
	{
	}


	USImageProperties::USImageProperties(const USImageProperties & a)
	{
		m_numScanlines = a.m_numScanlines;
		m_scanlineLayout = a.m_scanlineLayout;
		m_numSamples = a.m_numSamples;
		m_imageType = a.m_imageType;
		m_imageState = a.m_imageState;
		m_transducerType = a.m_transducerType;
		m_depth = a.m_depth;
		m_imageResolutionSet = a.m_imageResolutionSet;
		m_imageResolution = a.m_imageResolution;
		m_numChannels = a.m_numChannels;

		//copy specific parameter map
		m_specificParameters = a.m_specificParameters;

		//copy scanline info
		if (a.m_scanlines)
		{
			m_scanlines = a.m_scanlines;
		}
	}


	void USImageProperties::setImageType(USImageProperties::ImageType imageType)
	{
		m_imageType = imageType;
	}

	void USImageProperties::setImageState(USImageProperties::ImageState imageState)
	{
		m_imageState = imageState;
	}

	void USImageProperties::setTransducerType(USImageProperties::TransducerType transducerType)
	{
		m_transducerType = transducerType;
	}

	void USImageProperties::setScanlineLayout(vec2s scanlineLayout)
	{
		m_scanlineLayout = scanlineLayout;
		m_numScanlines = scanlineLayout.x * scanlineLayout.y;
	}

	void USImageProperties::setNumSamples(size_t numSamples)
	{
		m_numSamples = numSamples;
	}

	void USImageProperties::setNumChannels(size_t numChannels)
	{
		m_numChannels = numChannels;
	}

	void USImageProperties::setDepth(double depth)
	{
		m_depth = depth;
	}

	void USImageProperties::setImageResolution(double resolution)
	{
		m_imageResolution = resolution;
		m_imageResolutionSet = true;
	}


	void USImageProperties::setScanlineInfo(std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > scanlines)
	{
		m_scanlines = scanlines;
	}

	USImageProperties::ImageType USImageProperties::getImageType() const
	{
		return m_imageType;
	}

	USImageProperties::ImageState USImageProperties::getImageState() const
	{
		return m_imageState;
	}

	USImageProperties::TransducerType USImageProperties::getTransducerType() const
	{
		return m_transducerType;
	}

	size_t USImageProperties::getNumScanlines() const
	{
		return m_numScanlines;
	}

	vec2s USImageProperties::getScanlineLayout() const
	{
		return m_scanlineLayout;
	}

	size_t USImageProperties::getNumSamples() const
	{
		return m_numSamples;
	}

	size_t USImageProperties::getNumChannels() const
	{
		return m_numChannels;
	}

	double USImageProperties::getDepth() const
	{
		return m_depth;
	}



	std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > USImageProperties::getScanlineInfo() const
	{
		return m_scanlines;
	}

	bool USImageProperties::hasSpecificParameter(std::string parameterName) const
	{
		return m_specificParameters.find(parameterName) == m_specificParameters.end();
	}

	const std::string & USImageProperties::getSpecificParameter(std::string parameterName) const
	{
		//TODO test how it behaves if we access something not in the map (should result in an exception)
		return m_specificParameters.find(parameterName)->second;
	}

	const std::map<std::string, std::string>& USImageProperties::getSpecificParameters() const
	{
		return m_specificParameters;
	}

	double USImageProperties::getSampleDistance() const
	{
		return m_depth / (m_numSamples - 1);
	}

	double USImageProperties::getImageResolution() const
	{
		if (m_imageResolutionSet)
		{
			return m_imageResolution;
		}
		else {
			return m_depth / (m_numSamples - 1);
		}
	}

	bool USImageProperties::is2D() const
	{
		return m_scanlineLayout.x == 1 || m_scanlineLayout.y == 1;
	}
}
