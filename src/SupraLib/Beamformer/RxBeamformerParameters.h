// ================================================================================================
// 
// Copyright (C) 2017, Rüdiger Göbl - all rights reserved
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

#ifndef __RXBEAMFORMERPARAMETERS_H__
#define __RXBEAMFORMERPARAMETERS_H__

#include "USTransducer.h"
#include <memory>

namespace supra
{
	struct ScanlineRxParameters3D;
	class USRawData;

	class RxBeamformerParameters
	{
	public:
		typedef float LocationType;

		RxBeamformerParameters(
			std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > rxParameters,
			size_t numDepths,
			double depth,
			double speedOfSoundMMperS,
			const USTransducer* pTransducer);

		RxBeamformerParameters(
			size_t numRxScanlines,
			vec2s rxScanlineLayout,
			double speedOfSoundMMperS,
			const std::vector<LocationType> & rxDepths,
			const std::vector<ScanlineRxParameters3D> & rxScanlines,
			const std::vector<LocationType> & rxElementXs,
			const std::vector<LocationType> & rxElementYs,
			size_t rxNumDepths)
			: m_rxScanlineLayout(rxScanlineLayout)
			, m_numRxScanlines(numRxScanlines)
			, m_speedOfSoundMMperS(speedOfSoundMMperS)
			, m_rxNumDepths(rxNumDepths)
			, m_rxDepths(rxDepths)
			, m_rxScanlines(rxScanlines)
			, m_rxElementXs(rxElementXs)
			, m_rxElementYs(rxElementYs)
			, m_nonlinearElementToChannelMapping(false)
		{};

		size_t getNumRxScanlines() const { return m_numRxScanlines; }
		vec2s getRxScanlineLayout() const { return m_rxScanlineLayout; }
		double getSpeedOfSoundMMperS() const { return m_speedOfSoundMMperS; }
		const std::vector<LocationType> & getRxDepths() const { return m_rxDepths; }
		const std::vector<ScanlineRxParameters3D> & getRxScanlines() const { return m_rxScanlines; }
		const std::vector<LocationType> & getRxElementXs() const { return m_rxElementXs; }
		const std::vector<LocationType> & getRxElementYs() const { return m_rxElementYs; }
		bool getNonlinearElementToChannelMapping() const { return m_nonlinearElementToChannelMapping; }
		const std::vector<int32_t> & getElementToChannelMap() const { return m_elementToChannelMap; }
		size_t getRxNumDepths() const { return m_rxNumDepths; }


		bool operator== (const RxBeamformerParameters& b) const
		{
			return
				m_numRxScanlines == b.m_numRxScanlines &&
				m_rxScanlineLayout == b.m_rxScanlineLayout &&
				m_speedOfSoundMMperS == b.m_speedOfSoundMMperS &&
				m_rxDepths == b.m_rxDepths &&
				m_rxScanlines == b.m_rxScanlines &&
				m_rxElementXs == b.m_rxElementXs &&
				m_rxElementYs == b.m_rxElementYs &&
				m_rxNumDepths == b.m_rxNumDepths;
		}

		void writeMetaDataForMock(std::string filename, std::shared_ptr<const USRawData> rawData) const;
		static std::shared_ptr<USRawData> readMetaDataForMock(const std::string & mockMetadataFilename);
		static std::shared_ptr<USRawData> readMetaDataForMockAscii(const std::string & mockAsciiMetadataFilename);
		static std::shared_ptr<USRawData> readMetaDataForMockJson(const std::string & mockJsonMetadataFilename);
		
	private:

		size_t m_numRxScanlines;
		vec2s m_rxScanlineLayout;
		double m_speedOfSoundMMperS;
		std::vector<LocationType> m_rxDepths;
		std::vector<ScanlineRxParameters3D> m_rxScanlines;
		std::vector<LocationType> m_rxElementXs;
		std::vector<LocationType> m_rxElementYs;
		bool m_nonlinearElementToChannelMapping;
		std::vector<int32_t> m_elementToChannelMap;
		size_t m_rxNumDepths;
	};
}

#endif //!__RXBEAMFORMERPARAMETERS_H__
