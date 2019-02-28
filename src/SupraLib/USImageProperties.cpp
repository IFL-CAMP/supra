// ================================================================================================
// 
// Copyright (C) 2016, Rüdiger Göbl - all rights reserved
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
//          Rüdiger Göbl
//          Email r.goebl@tum.de
//          Chair for Computer Aided Medical Procedures
//          Technische Universität München
//          Boltzmannstr. 3, 85748 Garching b. München, Germany
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

#include "USImageProperties.h"

#include <json/json.h>

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
	{
	}


		USImageProperties::USImageProperties()
		: m_numScanlines(0)
		, m_scanlineLayout{ 0, 0 }
		, m_numSamples(0)
		, m_imageType(USImageProperties::ImageType::BMode)
		, m_imageState(USImageProperties::ImageState::Raw)
		, m_transducerType(USImageProperties::TransducerType::Linear)
		, m_depth(0.0)
		, m_imageResolutionSet(false)
		, m_imageResolution(0.0)
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

		//copy specific parameter map
		m_specificParameters = a.m_specificParameters;

		//copy scanline info
		if (a.m_scanlines)
		{
			m_scanlines = a.m_scanlines;
		}
	}

	USImageProperties::USImageProperties(const std::string & mockJsonMetadataFilename)
	{
		// open the 
		std::ifstream f(mockJsonMetadataFilename);
		if (!f.good())
		{
			logging::log_error("USImageProperties: Error opening json mock file ", mockJsonMetadataFilename);
			throw std::runtime_error("USImageProperties: Error opening json mock file");
		}

		// read and parse the json file
		Json::CharReaderBuilder builder;
		builder["collectComments"] = false;
		Json::Value jsonDoc;
		JSONCPP_STRING jsonErrs;
		bool jsonOk = parseFromStream(builder, f, &jsonDoc, &jsonErrs);
		f.close();

		if (!jsonOk)
		{
			logging::log_error("USImageProperties: Error parsing json mock file ", mockJsonMetadataFilename);
			logging::log_error("USImageProperties: Reason: ", jsonErrs);
			throw std::runtime_error("USImageProperties: Error parsing json mock file");
		}

		m_numSamples = jsonDoc["numSamples"].asLargestUInt();
		setScanlineLayout({ jsonDoc["scanlineLayout"]["x"].asLargestUInt(), jsonDoc["scanlineLayout"]["y"].asLargestUInt() });
		m_depth = jsonDoc["depth"].asDouble();

		m_imageType = static_cast<USImageProperties::ImageType>(jsonDoc["imageType"].asInt());
		m_imageState = static_cast<USImageProperties::ImageState>(jsonDoc["imageState"].asInt());
		m_transducerType = static_cast<USImageProperties::TransducerType>(jsonDoc["transducerType"].asInt());
		m_imageResolutionSet = jsonDoc["imageResolutionSet"].asBool();
		m_imageResolution = jsonDoc["imageResolution"].asDouble();

		m_scanlines =
			std::make_shared<std::vector<std::vector<ScanlineRxParameters3D> > >
			(m_scanlineLayout.x, std::vector<ScanlineRxParameters3D>(m_scanlineLayout.y));

		int scanlineIdx = 0;
		for (size_t idxY = 0; idxY < m_scanlineLayout.y; idxY++)
		{
			for (size_t idxX = 0; idxX < m_scanlineLayout.x; idxX++)
			{
				auto scanlineParams = jsonDoc["rxScanlines"][scanlineIdx];
				ScanlineRxParameters3D params;

				params.position = {
					scanlineParams["position"]["x"].asDouble(),
					scanlineParams["position"]["y"].asDouble(),
					scanlineParams["position"]["z"].asDouble()
				};
				params.direction = {
					scanlineParams["direction"]["x"].asDouble(),
					scanlineParams["direction"]["y"].asDouble(),
					scanlineParams["direction"]["z"].asDouble()
				};
				params.maxElementDistance = {
					scanlineParams["maxElementDistance"]["x"].asDouble(),
					scanlineParams["maxElementDistance"]["y"].asDouble()
				};

				for (int m = 0; m < 4; m++)
				{
					params.txParameters[m].firstActiveElementIndex = {
						static_cast<uint16_t>(scanlineParams["txParameters"][m]["firstActiveElementIndex"]["x"].asUInt()),
						static_cast<uint16_t>(scanlineParams["txParameters"][m]["firstActiveElementIndex"]["y"].asUInt())
					};
					params.txParameters[m].lastActiveElementIndex = {
						static_cast<uint16_t>(scanlineParams["txParameters"][m]["lastActiveElementIndex"]["x"].asUInt()),
						static_cast<uint16_t>(scanlineParams["txParameters"][m]["lastActiveElementIndex"]["y"].asUInt())
					};
					params.txParameters[m].txScanlineIdx = static_cast<uint16_t>(scanlineParams["txParameters"][m]["txScanlineIdx"].asUInt());
					params.txParameters[m].initialDelay = scanlineParams["txParameters"][m]["initialDelay"].asDouble();
					params.txWeights[m] = scanlineParams["txParameters"][m]["txWeights"].asDouble();
				}

				(*m_scanlines)[idxX][idxY] = params;
				scanlineIdx++;
			}
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

	void USImageProperties::writeMetaDataForMock(std::string filename) const
	{
		Json::Value jsonDoc;

		jsonDoc["numSamples"] = m_numSamples;
		jsonDoc["scanlineLayout"]["x"] = m_scanlineLayout.x;
		jsonDoc["scanlineLayout"]["y"] = m_scanlineLayout.y; 
		jsonDoc["depth"] = m_depth;

		jsonDoc["imageType"] = static_cast<int>(m_imageType);
		jsonDoc["imageState"] = static_cast<int>(m_imageState);
		jsonDoc["transducerType"]= static_cast<int>(m_transducerType);
		
		jsonDoc["imageResolutionSet"] = m_imageResolutionSet;
		jsonDoc["imageResolution"] = m_imageResolution;
		
		int scanlineIdx = 0;
		for (size_t idxY = 0; idxY < m_scanlineLayout.y; idxY++)
		{
			for (size_t idxX = 0; idxX < m_scanlineLayout.x; idxX++)
			{
				auto params = (*m_scanlines)[idxX][idxY];
				Json::Value scanlineParams;

				scanlineParams["position"]["x"] = params.position.x;
				scanlineParams["position"]["y"] = params.position.y;
				scanlineParams["position"]["z"] = params.position.z;

				scanlineParams["direction"]["x"] = params.direction.x;
				scanlineParams["direction"]["y"] = params.direction.y;
				scanlineParams["direction"]["z"] = params.direction.z;

				scanlineParams["maxElementDistance"]["x"] = params.maxElementDistance.x;
				scanlineParams["maxElementDistance"]["y"] = params.maxElementDistance.y;

				for (int m = 0; m < 4; m++)
				{
					scanlineParams["txParameters"][m]["firstActiveElementIndex"]["x"] = params.txParameters[m].firstActiveElementIndex.x;
					scanlineParams["txParameters"][m]["firstActiveElementIndex"]["y"] = params.txParameters[m].firstActiveElementIndex.y;

					scanlineParams["txParameters"][m]["lastActiveElementIndex"]["x"] = params.txParameters[m].lastActiveElementIndex.x;
					scanlineParams["txParameters"][m]["lastActiveElementIndex"]["y"] = params.txParameters[m].lastActiveElementIndex.y;

					scanlineParams["txParameters"][m]["txScanlineIdx"] = params.txParameters[m].txScanlineIdx;
					scanlineParams["txParameters"][m]["initialDelay"] = params.txParameters[m].initialDelay;
					scanlineParams["txParameters"][m]["txWeights"] = params.txWeights[m];
				}

				jsonDoc["rxScanlines"][scanlineIdx] = scanlineParams;
				
				scanlineIdx++;
			}
		}

		Json::StreamWriterBuilder wbuilder;
		wbuilder["indentation"] = "    ";

		std::ofstream f(filename);
		std::unique_ptr<Json::StreamWriter> writer(
			wbuilder.newStreamWriter());
		writer->write(jsonDoc, &f);
	}
}
