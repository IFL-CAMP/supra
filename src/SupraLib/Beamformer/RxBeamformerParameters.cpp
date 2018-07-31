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
		std::ofstream f(filename);
		f << "rawDataMockMetadata v 3" << std::endl;
		f << rawData->getNumElements() << " "
			<< rawData->getElementLayout().x << " "
			<< rawData->getElementLayout().y << " "
			<< rawData->getNumReceivedChannels() << " "
			<< rawData->getNumSamples() << " "
			<< rawData->getNumScanlines() << " "
			<< m_rxScanlineLayout.x << " "
			<< m_rxScanlineLayout.y << " "
			<< rawData->getImageProperties()->getDepth() << " "
			<< rawData->getSamplingFrequency() << " "
			<< m_rxNumDepths << " "
			<< m_speedOfSoundMMperS << std::endl;

		for (size_t idx = 0; idx < m_numRxScanlines; idx++)
		{
			f << m_rxScanlines[idx] << " ";
		}
		f << std::endl;
		for (size_t idx = 0; idx < m_rxNumDepths; idx++)
		{
			f << setprecision(9) << m_rxDepths[idx] << " ";
		}
		f << std::endl;
		for (size_t idx = 0; idx < rawData->getNumElements(); idx++)
		{
			f << setprecision(9) << m_rxElementXs[idx] << " ";
		}
		for (size_t idx = 0; idx < rawData->getNumElements(); idx++)
		{
			f << setprecision(9) << m_rxElementYs[idx] << " ";
		}
		f << std::endl;
		f.close();
	}

	std::shared_ptr<USRawData> RxBeamformerParameters::readMetaDataForMock(const std::string & mockMetadataFilename)
	{
		std::ifstream f(mockMetadataFilename);

		if (!f.good())
		{
			logging::log_error("RxBeamformerParameters: Error opening mock file ", mockMetadataFilename);
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
}