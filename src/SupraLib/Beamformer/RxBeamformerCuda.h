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

#ifndef __RXBEAMFORMERCUDA_H__
#define __RXBEAMFORMERCUDA_H__

#include "USTransducer.h"
#include "USImage.h"
#include "WindowFunction.h"

#include <memory>

namespace supra
{
	struct ScanlineRxParameters3D;
	template <typename T>
	class USRawData;

	using std::shared_ptr;

	class RxBeamformerCuda
	{
	public:
		RxBeamformerCuda(std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > rxParameters, size_t numDepths, double depth, double speedOfSoundMMperS, const USTransducer* pTransducer);
		~RxBeamformerCuda();

		// perform the receive beamforming
		template <typename ChannelDataType, typename ImageDataType>
		shared_ptr<USImage<ImageDataType> > performRxBeamforming(
			shared_ptr<const USRawData<ChannelDataType> > rawData,
			double fNumber,
			WindowType windowType,
			WindowFunction::ElementType windowParameters,
			bool interpolateBetweenTransmits,
			bool testSignal) const;

		template <typename ChannelDataType>
		void writeMetaDataForMock(std::string filename, shared_ptr<const USRawData<ChannelDataType> > rawData) const;
		template <typename ChannelDataType>
		static shared_ptr<USRawData<ChannelDataType> > readMetaDataForMock(const std::string & mockMetadataFilename);

	private:
		typedef float LocationType;

		//only for mock creation
		RxBeamformerCuda(
			size_t numRxScanlines,
			vec2s rxScanlineLayout,
			double speedOfSoundMMperS,
			const std::vector<LocationType> & rxDepths,
			const std::vector<ScanlineRxParameters3D> & rxScanlines,
			const std::vector<LocationType> & rxElementXs,
			const std::vector<LocationType> & rxElementYs,
			size_t rxNumDepths);

		void convertToDtSpace(double dt, size_t numTransducerElements) const;

		// Imaging parameters
		size_t m_numRxScanlines;
		vec2s m_rxScanlineLayout;
		double m_speedOfSoundMMperS;

		// prepared Rx parameters
		mutable std::unique_ptr<Container<LocationType> > m_pRxDepths;
		mutable std::unique_ptr<Container<ScanlineRxParameters3D> > m_pRxScanlines;
		mutable std::unique_ptr<Container<LocationType> > m_pRxElementXs;
		mutable std::unique_ptr<Container<LocationType> > m_pRxElementYs;
		size_t m_rxNumDepths;

		mutable double m_lastSeenDt;
		mutable shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		mutable shared_ptr<const USImageProperties> m_editedImageProperties;

		bool m_is3D;

		static constexpr size_t m_windowFunctionNumEntries = 64;
		mutable std::unique_ptr<WindowFunction> m_windowFunction;
	};
}

#endif //!__RXBEAMFORMERCUDA_H__
