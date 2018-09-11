// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2017, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "RxBeamformerParameters.h"
#include "USRawData.h"

#include <json/json.h>

#include <memory>
#include <string>
#include <iomanip>

namespace supra
{
	RxBeamformerParameters::RxBeamformerParameters(
		std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > rxParameters,
		size_t numDepths,
		double depth,
		double speedOfSoundMMperS,
		const USTransducer* pTransducer)
		: m_speedOfSoundMMperS(speedOfSoundMMperS)
		, m_rxNumDepths(numDepths)
	{
		vec2s numRxScanlines = { rxParameters->size(), (*rxParameters)[0].size() };
		m_numRxScanlines = numRxScanlines.x*numRxScanlines.y;
		m_rxScanlineLayout = numRxScanlines;
		size_t numElements = pTransducer->getNumElements();

		// create and fill new buffers
		m_rxDepths = std::vector<LocationType>(numDepths);

		m_rxScanlines = std::vector<ScanlineRxParameters3D>(m_numRxScanlines);

		m_rxElementXs = std::vector<LocationType>(numElements);
		m_rxElementYs = std::vector<LocationType>(numElements);

		for (size_t zIdx = 0; zIdx < numDepths; zIdx++)
		{
			m_rxDepths[zIdx] = static_cast<LocationType>(zIdx*depth / (numDepths - 1));
		}
		size_t scanlineIdx = 0;
		for (size_t yIdx = 0; yIdx < numRxScanlines.y; yIdx++)
		{
			for (size_t xIdx = 0; xIdx < numRxScanlines.x; xIdx++)
			{
				auto s = (*rxParameters)[xIdx][yIdx];
				m_rxScanlines[scanlineIdx] = s;
				scanlineIdx++;
			}
		}

		auto centers = pTransducer->getElementCenterPoints();
		for (size_t x_elemIdx = 0; x_elemIdx < numElements; x_elemIdx++)
		{
			m_rxElementXs[x_elemIdx] = static_cast<LocationType>(centers->at(x_elemIdx).x);
			m_rxElementYs[x_elemIdx] = static_cast<LocationType>(centers->at(x_elemIdx).y);
		}
	}

	using std::setw;
	using std::setprecision;

	void RxBeamformerParameters::writeMetaDataForMock(std::string filename, std::shared_ptr<const USRawData> rawData) const
	{
		Json::Value jsonDoc;

		jsonDoc["numElements"] = rawData->getNumElements();
		jsonDoc["numReceivedChannels"] = rawData->getNumReceivedChannels();
		jsonDoc["numSamples"] = rawData->getNumSamples();
		jsonDoc["numTxScanlines"] = rawData->getNumScanlines();
		
		jsonDoc["scanlineLayout"]["x"] = m_rxScanlineLayout.x;
		jsonDoc["scanlineLayout"]["y"] = m_rxScanlineLayout.y;
		jsonDoc["elementLayout"]["x"] = rawData->getElementLayout().x;
		jsonDoc["elementLayout"]["y"] = rawData->getElementLayout().y;
		jsonDoc["depth"] = rawData->getImageProperties()->getDepth();
		jsonDoc["samplingFrequency"] = rawData->getSamplingFrequency();
		jsonDoc["rxNumDepths"] = m_rxNumDepths;
		jsonDoc["speedOfSoundMMperS"] = m_speedOfSoundMMperS;
		
		for (int idx = 0; idx < m_numRxScanlines; idx++)
		{
			auto params = m_rxScanlines[idx];
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

			jsonDoc["rxScanlines"][idx] = scanlineParams;
		}
		for (int idx = 0; idx < rawData->getNumElements(); idx++)
		{
			jsonDoc["rxElementPosition"][idx]["x"] = m_rxElementXs[idx];
			jsonDoc["rxElementPosition"][idx]["y"] = m_rxElementYs[idx];

		}

		Json::StreamWriterBuilder wbuilder;
		wbuilder["indentation"] = "    ";

		std::ofstream f(filename);
		std::unique_ptr<Json::StreamWriter> writer(
			wbuilder.newStreamWriter());
		writer->write(jsonDoc, &f);
	}

	std::shared_ptr<USRawData> RxBeamformerParameters::readMetaDataForMock(const std::string & mockMetadataFilename)
	{
		std::shared_ptr<USRawData> protoRawData;

		const std::string suffixAscii{ ".mock" };
		const std::string suffixJson{ ".json" };

		if (mockMetadataFilename.compare(
			mockMetadataFilename.length() - suffixAscii.length(),
			suffixAscii.length(),
			suffixAscii) == 0)
		{
			protoRawData = readMetaDataForMockAscii(mockMetadataFilename);
		}
		else if (mockMetadataFilename.compare(
			mockMetadataFilename.length() - suffixJson.length(),
			suffixJson.length(),
			suffixJson) == 0)
		{
			protoRawData = readMetaDataForMockJson(mockMetadataFilename);
		}
		else
		{
			logging::log_error("RxBeamformerParameters: Error opening mock file ", mockMetadataFilename);
			return nullptr;
		}

		return protoRawData;
	}

	std::shared_ptr<USRawData> RxBeamformerParameters::readMetaDataForMockAscii(const std::string & mockAsciiMetadataFilename)
	{
		std::ifstream f(mockAsciiMetadataFilename);

		if (!f.good())
		{
			logging::log_error("RxBeamformerParameters: Error opening ascii mock file ", mockAsciiMetadataFilename);
			return nullptr;
		}

		std::shared_ptr<USRawData> rawData;

		size_t numElements;
		size_t numReceivedChannels;
		size_t numSamples;
		size_t numTxScanlines;
		vec2s scanlineLayout;
		vec2s elementLayout;
		double depth;
		double samplingFrequency;
		size_t rxNumDepths;
		double speedOfSoundMMperS;
		//f << "rawDataMockMetadata v 1";
		std::string dummy;
		int version;
		f >> dummy;
		f >> dummy;
		f >> version;

		f >> numElements;
		f >> elementLayout.x;
		f >> elementLayout.y;
		f >> numReceivedChannels;
		f >> numSamples;
		f >> numTxScanlines;
		f >> scanlineLayout.x;
		f >> scanlineLayout.y;
		f >> depth;
		f >> samplingFrequency;
		f >> rxNumDepths;
		f >> speedOfSoundMMperS;

		size_t numRxScanlines = scanlineLayout.x*scanlineLayout.y;

		std::vector<ScanlineRxParameters3D> rxScanlines(numRxScanlines);
		std::vector<LocationType> rxDepths(rxNumDepths);
		std::vector<LocationType> rxElementXs(numElements);
		std::vector<LocationType> rxElementYs(numElements);

		std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > scanlines =
			std::make_shared<std::vector<std::vector<ScanlineRxParameters3D> > >
				(scanlineLayout.x, std::vector<ScanlineRxParameters3D>(scanlineLayout.y));

		size_t scanlineIdx = 0;
		for (size_t idxY = 0; idxY < scanlineLayout.y; idxY++)
		{
			for (size_t idxX = 0; idxX < scanlineLayout.x; idxX++)
			{
				ScanlineRxParameters3D params;
				f >> params;
				(*scanlines)[idxX][idxY] = params;
				rxScanlines[scanlineIdx] = params;
				scanlineIdx++;
			}
		}

		for (size_t idx = 0; idx < rxNumDepths; idx++)
		{
			LocationType val;
			f >> val;
			rxDepths[idx] = val;
		}
		for (size_t idx = 0; idx < numElements; idx++)
		{
			LocationType val;
			f >> val;
			rxElementXs[idx] = val;
		}
		for (size_t idx = 0; idx < numElements; idx++)
		{
			LocationType val;
			f >> val;
			rxElementYs[idx] = val;
		}

		f.close();

		auto imageProps = std::make_shared<USImageProperties>(
			vec2s{ numTxScanlines, 1 },
			rxNumDepths,
			USImageProperties::ImageType::BMode,
			USImageProperties::ImageState::Raw,
			USImageProperties::TransducerType::Linear,
			depth);

		imageProps->setScanlineInfo(scanlines);

		auto rxBeamformer = std::shared_ptr<RxBeamformerParameters>(new RxBeamformerParameters(
			numRxScanlines,
			scanlineLayout,
			speedOfSoundMMperS,
			rxDepths,
			rxScanlines,
			rxElementXs,
			rxElementYs,
			rxNumDepths));

		auto pRawData = std::make_shared<USRawData>(
			numTxScanlines,
			numElements,
			elementLayout,
			numReceivedChannels,
			numSamples,
			samplingFrequency,
			nullptr,
			rxBeamformer,
			imageProps,
			0,
			0);

		return pRawData;
	}

	std::shared_ptr<USRawData> RxBeamformerParameters::readMetaDataForMockJson(const std::string & mockJsonMetadataFilename)
	{
		// open the 
		std::ifstream f(mockJsonMetadataFilename);
		if (!f.good())
		{
			logging::log_error("RxBeamformerParameters: Error opening json mock file ", mockJsonMetadataFilename);
			return nullptr;
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
			logging::log_error("RxBeamformerParameters: Error parsing json mock file ", mockJsonMetadataFilename);
			logging::log_error("RxBeamformerParameters: Reason: ", jsonErrs);
			return nullptr;
		}


		std::shared_ptr<USRawData> rawData;

		size_t numElements = jsonDoc["numElements"].asLargestUInt();
		size_t numReceivedChannels = jsonDoc["numReceivedChannels"].asLargestUInt();
		size_t numSamples = jsonDoc["numSamples"].asLargestUInt();
		size_t numTxScanlines = jsonDoc["numTxScanlines"].asLargestUInt();
		vec2s scanlineLayout{ jsonDoc["scanlineLayout"]["x"].asLargestUInt(), jsonDoc["scanlineLayout"]["y"].asLargestUInt() };
		vec2s elementLayout{ jsonDoc["elementLayout"]["x"].asLargestUInt(), jsonDoc["elementLayout"]["y"].asLargestUInt() };
		double depth = jsonDoc["depth"].asDouble();
		double samplingFrequency = jsonDoc["samplingFrequency"].asDouble();
		size_t rxNumDepths = jsonDoc["rxNumDepths"].asLargestUInt();
		double speedOfSoundMMperS = jsonDoc["speedOfSoundMMperS"].asDouble();
		
		size_t numRxScanlines = scanlineLayout.x*scanlineLayout.y;

		std::vector<ScanlineRxParameters3D> rxScanlines(numRxScanlines);
		std::vector<LocationType> rxDepths(rxNumDepths);
		std::vector<LocationType> rxElementXs(numElements);
		std::vector<LocationType> rxElementYs(numElements);

		std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > scanlines =
			std::make_shared<std::vector<std::vector<ScanlineRxParameters3D> > >
			(scanlineLayout.x, std::vector<ScanlineRxParameters3D>(scanlineLayout.y));

		int scanlineIdx = 0;
		for (size_t idxY = 0; idxY < scanlineLayout.y; idxY++)
		{
			for (size_t idxX = 0; idxX < scanlineLayout.x; idxX++)
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

				(*scanlines)[idxX][idxY] = params;
				rxScanlines[scanlineIdx] = params;
				scanlineIdx++;
			}
		}

		for (int idx = 0; idx < rxNumDepths; idx++)
		{
			// not stored in the json, but we can deduce it
			rxDepths[idx] = static_cast<float>(depth * static_cast<double>(idx) / (rxNumDepths - 1));
		}
		for (int idx = 0; idx < numElements; idx++)
		{
			rxElementXs[idx] = jsonDoc["rxElementPosition"][idx]["x"].asFloat();
			rxElementYs[idx] = jsonDoc["rxElementPosition"][idx]["y"].asFloat();
		}
		
		auto imageProps = std::make_shared<USImageProperties>(
			vec2s{ numTxScanlines, 1 },
			rxNumDepths,
			USImageProperties::ImageType::BMode,
			USImageProperties::ImageState::Raw,
			USImageProperties::TransducerType::Linear,
			depth);

		imageProps->setScanlineInfo(scanlines);

		auto rxBeamformer = std::shared_ptr<RxBeamformerParameters>(new RxBeamformerParameters(
			numRxScanlines,
			scanlineLayout,
			speedOfSoundMMperS,
			rxDepths,
			rxScanlines,
			rxElementXs,
			rxElementYs,
			rxNumDepths));

		auto pRawData = std::make_shared<USRawData>(
			numTxScanlines,
			numElements,
			elementLayout,
			numReceivedChannels,
			numSamples,
			samplingFrequency,
			nullptr,
			rxBeamformer,
			imageProps,
			0,
			0);

		return pRawData;
	}
}