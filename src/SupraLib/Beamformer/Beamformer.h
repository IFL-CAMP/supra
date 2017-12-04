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

#ifndef __BEAMFORMER_H__
#define __BEAMFORMER_H__

#include "USTransducer.h"
#include "USImage.h"
#include "RxBeamformerCuda.h"
#include "WindowFunction.h"

#include <memory>

namespace supra
{
	// The transmit parameters for one scanline
	struct ScanlineTxParameters3D
	{
		vec2s firstActiveElementIndex;	// index of the first active transducer element
		vec2s lastActiveElementIndex;	// index of the last active transducer element
		rect2s txAperture;				// contains the (sub)aperture to transmit on
		vec position;	                // the position of the scanline
		vec direction;                  // direction of the scanline
		double maxDelay;				// the largest applied delay in [s]. This defines the start of the part of the acquisition we are interested in
		std::vector<std::vector<bool> >   elementMap;   // elements that have to be recieved on
		std::vector<std::vector<bool> >   txElementMap; // elements that have to be fired
		std::vector<std::vector<double> > delays;       // delays in [s] that have to be applied per ACTIVE element  (delays.size() == nnz(elementMap))
		std::vector<std::vector<double> > weights;      // relative weights [0...1] to use per ACTIVE element during transmission (weights.size() == nnz(elementMap))

		operator ScanlineRxParameters3D::TransmitParameters() const
		{
			ScanlineRxParameters3D::TransmitParameters ret;
			ret.firstActiveElementIndex = static_cast<vec2T<uint16_t>>(firstActiveElementIndex);
			ret.lastActiveElementIndex = static_cast<vec2T<uint16_t>>(lastActiveElementIndex);
			ret.initialDelay = maxDelay;
			return ret;
		}
	};

	class Beamformer
	{
	public:
		enum ScanType {
			Linear,
			Phased,
			Biphased,
			Planewave
		};

		Beamformer();
		~Beamformer();

		Beamformer(const std::shared_ptr<Beamformer> bf);


		void setTransducer(const USTransducer* transducer);

		void setScanType(const std::string scanType);

		/// Set speed of sound as assumed for beamforming [m/s]
		void setSpeedOfSound(const double speedOfSound);

		/// Set penetration depth [mm]
		void setDepth(const double depth);

		/// Set number of transmit scanlines in x/y 
		void setNumScanlines(const vec2s numScanlines);

		/// Set subdivisiion from transmit to receive scanlines x/y 
		void setRxScanlineSubdivision(const vec2s scanlineSubdivision);

		/// Set window type for transmit aperture (apodization)
		// Can be Hann, Hamming, Rectangular Gauss
		void setTxWindowType(const std::string windowType);

		/// Set associated window parameters for selected window type
		void setWindowParameter(const WindowFunction::ElementType windowParameter);

		/// Set field of view for scanline opening [degrees x/y] 
		void setFov(const vec2 fov); 

		// set center steering angle for scanline opening [degrees x/y]
		void setTxSteeringAngle(const vec2 txSteeringAngle);

		/// Specify the maximum aperture to be used for beamforming [channels x/y] 
		void setMaxApertureSize (const vec2s aptertureSize);

		/// Specify the maximum transmit aperture to be used for beamforming [channels x/y] 
		void setTxMaxApertureSize (const vec2s txApertureSize);

		/// Activate transmit focusing
		void setTxFocusActive(const bool txFocusActive);

		/// Set transmit focus depth [mm] 
		void setTxFocusDepth(const double txFocusDepth);

		/// Set transmit focus width [mm]
		void setTxFocusWidth(const double txFocusWidth);

		/// Activate correction for speed of sound in matching layers
        void setTxCorrectMatchingLayers(const bool txCorrectMatchingLayers);

		/// Set receive focusing depth [mm]
		void setRxFocusDepth(const double rxFocusDepth);

		/// Set number or discrete receive focus steps to be calculated
		void setNumDepths(const size_t numDepths);


		std::string getScanType() const;
		double getSpeedOfSound() const;
		double getDepth() const;
		vec2s getNumScanlines() const;
		vec2s getRxScanlineSubdivision() const;
		vec2s getNumRxScanlines() const;
		vec2s getMaxActiveElements() const;
		vec2s getMaxTxElements() const;
		vec2 getFov() const;
		vec2s getApertureSize () const;
		vec2s getTxApertureSize () const;
		bool getTxFocusActive() const;
		double getTxFocusDepth() const;
		double getTxFocusWidth() const;
        bool getTxCorrectMatchingLayers() const;
		vec2 getTxSteeringAngle() const;
		double getRxFocusDepth() const;
		size_t getNumDepths() const;

		

		vec2s getScanlineLayout() const;

		bool isReady() const;

		// prepare and get the transmit parameters for the image setting
		void computeTxParameters();
		const std::vector<ScanlineTxParameters3D>* getTxParameters();
		std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > getRxParameters();

		// prepare and perform the receive beamforming
		std::shared_ptr<const RxBeamformerCuda> getCurrentRxBeamformer();

	private:
		typedef float LocationType;
		ScanlineTxParameters3D getTxScanline3D(
			rect2s activeAperture,
			rect2s txAperture,
			vec2d scanlineStart,
			vec2d steeringAngle);
		ScanlineRxParameters3D getRxScanline3D(size_t txScanlineIdx, const ScanlineTxParameters3D& txScanline);
		ScanlineRxParameters3D getRxScanline3DInterpolated(
			size_t txScanlineIdx1, const ScanlineTxParameters3D& txScanline1,
			size_t txScanlineIdx2, const ScanlineTxParameters3D& txScanline2,
			size_t txScanlineIdx3, const ScanlineTxParameters3D& txScanline3,
			size_t txScanlineIdx4, const ScanlineTxParameters3D& txScanline4,
			vec2 interp);
		static rect2s computeAperture(vec2s layout, vec2s apertureSize, vec2 relativePosition);

		std::vector<ScanlineTxParameters3D> m_txParameters;
		std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > m_rxParameters;
		std::shared_ptr<const RxBeamformerCuda> m_pRxBeamformer;

		const USTransducer* m_pTransducer;

		// Imaging parameters
		ScanType m_type;
		bool m_correctMatchingLayers;

		vec2s m_numScanlines;
		vec2s m_rxScanlineSubdivision;
		vec2s m_numRxScanlines;     // updated internally
		vec2s m_maxApertureSize;
		vec2s m_txMaxApertureSize;

		WindowType m_txWindow;
		WindowFunction::ElementType m_txWindowParameter;
		double m_depth;
		vec2 m_fov;
		bool m_txFocusActive;
		double m_txFocusDepth;
		double m_txFocusWidth;
		double m_rxFocusDepth;
		double m_speedOfSound;
		double m_speedOfSoundMMperS; 	// updated internally
		vec2 m_txSteeringAngle;
		
		//double m_fNumber;
		uint32_t m_numSamplesRecon;


		bool m_ready;				// state if beamformers internal state is fully defined (all parameters calculated)
	};
}

#endif //!__BEAMFORMER_H__
