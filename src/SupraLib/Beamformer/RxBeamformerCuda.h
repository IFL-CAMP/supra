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

#include "USImage.h"
#include "WindowFunction.h"
#include "RxBeamformerParameters.h"

#include <memory>

namespace supra
{
	struct ScanlineRxParameters3D;
	class USRawData;

	using std::shared_ptr;

	class RxBeamformerCuda
	{
	public:
		enum RxSampleBeamformer {
			DelayAndSum,
			DelayAndStdDev,
			TestSignal,
			INVALID
		};

		RxBeamformerCuda(const RxBeamformerParameters& parameters);
		~RxBeamformerCuda();

		// perform the receive beamforming
		template <typename ChannelDataType, typename ImageDataType>
		shared_ptr<USImage> performRxBeamforming(
			RxSampleBeamformer sampleBeamformer,
			shared_ptr<const USRawData> rawData,
			double fNumber,
			double speedOfSoundMMperS,
			WindowType windowType,
			WindowFunction::ElementType windowParameters,
			bool interpolateBetweenTransmits,
			int32_t additionalOffset) const;

	private:
		typedef RxBeamformerParameters::LocationType LocationType;

		void convertToDtSpace(double dt, double speedOfSoundMMperS, size_t numTransducerElements) const;

		// Imaging parameters
		size_t m_numRxScanlines;
		vec2s m_rxScanlineLayout;
		std::unique_ptr<Container<int32_t> > m_elementToChannelMap;

		// prepared Rx parameters
		mutable std::unique_ptr<Container<LocationType> > m_pRxDepths;
		mutable std::unique_ptr<Container<ScanlineRxParameters3D> > m_pRxScanlines;
		mutable std::unique_ptr<Container<LocationType> > m_pRxElementXs;
		mutable std::unique_ptr<Container<LocationType> > m_pRxElementYs;
		size_t m_rxNumDepths;

		mutable double m_speedOfSoundMMperS;
		mutable double m_lastSeenDt;
		mutable shared_ptr<const USImageProperties> m_lastSeenImageProperties;
		mutable shared_ptr<const USImageProperties> m_editedImageProperties;

		bool m_is3D;

		static constexpr size_t m_windowFunctionNumEntries = 64;
		mutable std::unique_ptr<WindowFunction> m_windowFunction;
	};
}

#endif //!__RXBEAMFORMERCUDA_H__
