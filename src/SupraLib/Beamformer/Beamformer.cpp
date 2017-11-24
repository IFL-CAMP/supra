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

#include "Beamformer.h"
#include "RxBeamformerCuda.h"

#include <exception>
#include <cassert>
#include <algorithm>
#include <memory>
#include <utilities/Logging.h>

namespace supra
{
	using std::max;
	using std::round;
	using std::vector;
	using std::make_shared;
	using std::tuple;

	using namespace logging;

	Beamformer::Beamformer()
		: m_type(Beamformer::Linear)
		, m_correctMatchingLayers(true)
		, m_numScanlines{ 0, 0 }
		, m_rxScanlineSubdivision{ 1, 1 }
		, m_numRxScanlines{ 0, 0 }
		, m_txWindow(WindowRectangular)
		, m_depth(0.0)
		, m_txFocusActive(false)
		, m_txFocusDepth(0.0)
		, m_txFocusWidth(0.0)
		, m_rxFocusDepth(0.0)
		, m_speedOfSound(1540.0)
		, m_speedOfSoundMMperS(m_speedOfSound * 1000.0)
		, m_fov{ 0.0, 0.0 }
		, m_pTransducer(nullptr)
		, m_pRxBeamformer(nullptr)
		, m_txSteeringAngle{0,0}
		, m_ready(false)
	{
	}

	Beamformer::Beamformer(const std::shared_ptr<Beamformer> bf)
		: m_txParameters(bf->m_txParameters)
		, m_rxParameters(bf->m_rxParameters)
		, m_pRxBeamformer(bf->m_pRxBeamformer)
		, m_pTransducer(bf->m_pTransducer)
		, m_type(bf->m_type)
		, m_numScanlines(bf->m_numScanlines)
		, m_rxScanlineSubdivision(bf->m_rxScanlineSubdivision)
		, m_numRxScanlines(bf->m_numRxScanlines)
		, m_maxApertureSize(bf->m_maxApertureSize)
		, m_txMaxApertureSize(bf->m_txMaxApertureSize)
		, m_fov(bf->m_fov)
		, m_depth(bf->m_depth)
		, m_txFocusActive(bf->m_txFocusActive)
		, m_txFocusDepth(bf->m_txFocusDepth)
		, m_txFocusWidth(bf->m_txFocusWidth)
		, m_rxFocusDepth(bf->m_rxFocusDepth)
		, m_speedOfSound(bf->m_speedOfSound)
		, m_speedOfSoundMMperS(bf->m_speedOfSoundMMperS)
		, m_txSteeringAngle(bf->m_txSteeringAngle)
		, m_numSamplesRecon(bf->m_numSamplesRecon)
		, m_ready(false)	

	{

	}

	Beamformer::~Beamformer()
	{
	}

	void Beamformer::setTransducer(const USTransducer* transducer)
	{
		if (transducer != m_pTransducer)
		{
			m_ready = false;
		}

		m_pTransducer = transducer;
	}

	
	void Beamformer::setScanType(const std::string scanType)
	{
		if (scanType == "linear") {
			m_type = Beamformer::Linear;
		} else if (scanType == "phased") {
			m_type = Beamformer::Phased;
		} else if (scanType == "biphased") {
			m_type = Beamformer::Biphased;
		} else if (scanType == "planewave") {
			m_type = Beamformer::Planewave;
		} else { 
			throw std::invalid_argument("Scan type string invalid");
		}

		m_ready = false;
	}

	void Beamformer::setSpeedOfSound(const double speedOfSound)
	{
		if (speedOfSound != m_speedOfSound)
		{
			m_ready = false;
		}

		m_speedOfSound = speedOfSound;
		m_speedOfSoundMMperS = (m_speedOfSound * 1000.0);
	}

	void Beamformer::setDepth(const double depth)
	{
		if (depth != m_depth)
		{
			m_ready = false;
		}
		m_depth = depth;
	}

	void Beamformer::setNumScanlines(const vec2s numScanlines)
	{
		if (numScanlines != m_numScanlines)
		{
			m_ready = false;
		}
		m_numScanlines = numScanlines;
	}

	void Beamformer::setRxScanlineSubdivision(const vec2s rxScanlineSubdivision)
	{
		if (rxScanlineSubdivision != m_rxScanlineSubdivision)
		{
			m_ready = false;
		}
		m_rxScanlineSubdivision = rxScanlineSubdivision;
		m_numRxScanlines = {
			m_numScanlines.x + (m_rxScanlineSubdivision.x - 1)*(m_numScanlines.x - 1),
			m_numScanlines.y + (m_rxScanlineSubdivision.y - 1)*(m_numScanlines.y - 1) };
	}


	void Beamformer::setFov(const vec2 fov)
	{
		auto fovRad = fov / 180.0 * M_PI;
		if (fovRad != m_fov)
		{
			m_ready = false;
		}
		m_fov = fovRad;
	}

	void Beamformer::setMaxApertureSize (const vec2s apertureSize)
	{
		if (apertureSize != m_maxApertureSize)
		{
			m_ready = false;
		}
		m_maxApertureSize = apertureSize;
	}

	void Beamformer::setTxMaxApertureSize (const vec2s txApertureSize)
	{
		if (txApertureSize != m_txMaxApertureSize)
		{
			m_ready = false;
		}
		m_txMaxApertureSize = txApertureSize;
	}

	void Beamformer::setTxWindowType(const std::string windowType)
	{
		WindowType selectedTxWindowType = WindowRectangular;
		if (windowType == "Hann")
		{
			selectedTxWindowType = WindowHann;
		}
		if (windowType == "Hamming")
		{
			selectedTxWindowType = WindowHamming;
		}
		if (windowType == "Rectangular")
		{
			selectedTxWindowType = WindowRectangular;
		}
		if (windowType == "Gauss")
		{
			selectedTxWindowType = WindowGauss;
		}


		if (selectedTxWindowType != m_txWindow)
		{
			m_ready = false;
			m_txWindow = selectedTxWindowType;
		}
	}

	void Beamformer::setWindowParameter(const WindowFunction::ElementType winParameter)
	{
		if (winParameter != m_txWindowParameter)
		{
			m_txWindowParameter = winParameter;
			m_ready = false;
		}
	}

	void Beamformer::setTxFocusActive(const bool txFocusActive)
	{
		if (txFocusActive != m_txFocusActive)
		{
			m_ready = false;
		}

		m_txFocusActive = txFocusActive;
	}

	void Beamformer::setTxFocusDepth(const double txFocusDepth)
	{
		if (txFocusDepth != m_txFocusDepth)
		{
			m_ready = false;
		}
		m_txFocusDepth = txFocusDepth;
		
	}

	void Beamformer::setTxFocusWidth(const double txFocusWidth)
	{
		if (txFocusWidth != m_txFocusWidth)
		{
			m_ready = false;
		}
		m_txFocusWidth = txFocusWidth;
	}

	void Beamformer::setTxCorrectMatchingLayers(const bool correctMatchingLayers)
	{
		if (correctMatchingLayers != m_correctMatchingLayers)
		{
			m_ready = false;
		}
		m_correctMatchingLayers = correctMatchingLayers;
	}

	void Beamformer::setTxSteeringAngle(const vec2 txSteeringAngle)
	{
		if (txSteeringAngle != m_txSteeringAngle)
		{
			m_ready = false;
		}
		m_txSteeringAngle = txSteeringAngle;
	}

	void Beamformer::setRxFocusDepth(const double rxFocusDepth)
	{
		if (rxFocusDepth != m_rxFocusDepth)
		{
			m_ready = false;
		}
		m_rxFocusDepth = rxFocusDepth;
	}

	void Beamformer::setNumDepths(size_t numDepths)
	{
		m_numSamplesRecon = static_cast<uint32_t>(numDepths);
		m_ready = false;
	}

	
	std::string Beamformer::getScanType() const
	{
		std::string scanType;
		if (m_type == Beamformer::Linear) {
			scanType = "linear";
		} else if (m_type == Beamformer::Phased) {
			scanType = "phased";
		} else if (m_type == Beamformer::Biphased) {
			scanType == "biphased";
		} else if (m_type == Beamformer::Planewave) {
			scanType == "planewave";
		} else { 
			throw std::invalid_argument("Scan type string invalid");
		}
		return scanType;
	}

	double Beamformer::getSpeedOfSound() const
	{
		return m_speedOfSound;
	}

	double Beamformer::getDepth() const
	{
		return m_depth;
	}


	vec2s Beamformer::getNumScanlines() const
	{
		return m_numScanlines;
	}

	vec2s Beamformer::getRxScanlineSubdivision() const
	{
		return m_rxScanlineSubdivision;
	}

	vec2s Beamformer::getNumRxScanlines() const
	{
		return m_numRxScanlines;
	}

	vec2 Beamformer::getFov() const
	{
		return m_fov / M_PI * 180.0;
	}

	vec2s Beamformer::getApertureSize () const
	{
		return m_maxApertureSize;
	}

	vec2s Beamformer::getTxApertureSize () const
	{
		return m_txMaxApertureSize;
	}

	bool Beamformer::getTxFocusActive() const
	{
		return m_txFocusActive;
	}

	double Beamformer::getTxFocusDepth() const
	{
		return m_txFocusDepth;
	}

	double Beamformer::getTxFocusWidth() const
	{
		return m_txFocusWidth;
	}

	bool Beamformer::getTxCorrectMatchingLayers() const
	{
		return m_correctMatchingLayers;
	}

	vec2 Beamformer::getTxSteeringAngle() const
	{
		return m_txSteeringAngle;
	}

	double Beamformer::getRxFocusDepth() const
	{
		return m_rxFocusDepth;
	}

	size_t Beamformer::getNumDepths() const
	{
		return m_numSamplesRecon;
	}


	bool Beamformer::isReady() const
	{
		return m_ready;
	}
	
	void Beamformer::computeTxParameters()
	{
		if (m_pTransducer &&
			m_numScanlines.x > 0 && m_numScanlines.y > 0 &&
			m_maxApertureSize.x > 0 && m_maxApertureSize.y > 0 &&
			m_depth > 0 &&
			m_txFocusDepth >= 0 &&
			m_rxFocusDepth >= 0 &&
			m_speedOfSound > 0)
		{
			size_t numBeamsTotal = m_numScanlines.x * m_numScanlines.y; 
			logging::log_info("Beamformer: computing TX parameters");

			m_txParameters.clear();
			m_txParameters.resize(numBeamsTotal);

			vec2s elementLayout = m_pTransducer->getElementLayout();

			if (m_type == Linear)
			{
				size_t numScanlines = m_numScanlines.x;
				auto elementCenterpoints = m_pTransducer->getElementCenterPoints();
				vec firstElement = elementCenterpoints->at(0);
				vec lastElement = elementCenterpoints->at(m_pTransducer->getNumElements() - 1);


				double angleRad = m_txSteeringAngle.x * M_PI / 180.0;

				
				//evenly space the scanlines between the first and last element
				for (size_t scanlineIdx = 0; scanlineIdx < numScanlines; scanlineIdx++)
				{
					// the position of the scanline on the x axis
					double scanlinePosition = firstElement.x +
						static_cast<double>(scanlineIdx) / (numScanlines - 1) * (lastElement.x - firstElement.x);
					
					// the scanline position in terms of elementIndices
					double scanlinePositionRelative = static_cast<double>(scanlineIdx) / (numScanlines - 1) * (m_pTransducer->getNumElements() - 1);

					rect2s activeAperture = computeAperture(elementLayout, m_maxApertureSize, { scanlinePositionRelative, 0 });
					rect2s txAperture = computeAperture(elementLayout, m_txMaxApertureSize, { scanlinePositionRelative, 0 });

					m_txParameters[scanlineIdx] = getTxScanline3D(activeAperture, txAperture, vec2d{ scanlinePosition, 0 }, vec2d{ angleRad, 0 });
				}
		
			}
			else if (m_type == Phased)
			{
				//TODO right now full aperture
				assert(m_pTransducer->getNumElements() == m_maxApertureSize.x);

				size_t numScanlines = m_numScanlines.x;
				double fov = m_fov.x;

				auto elementCenterpoints = m_pTransducer->getElementCenterPoints();

				double scanlineStartX;
				size_t numElements = m_pTransducer->getNumElements();
				if (numElements % 2 == 1)
				{
					//15 -> (15-1)/2
					scanlineStartX = elementCenterpoints->at((numElements - 1) / 2).x;
				}
				else {
					//16 -> (16/2 - 1) & (16/2)
					scanlineStartX = (elementCenterpoints->at(numElements / 2 - 1).x +
						elementCenterpoints->at(numElements / 2).x) / 2;
				}

				rect2s activeAperture = computeAperture(elementLayout, m_maxApertureSize, vec2d{ (static_cast<double>(numElements) - 1) / 2, 0 });
				rect2s txAperture = computeAperture(elementLayout, m_txMaxApertureSize, vec2d{ (static_cast<double>(numElements) - 1) / 2, 0 });

				for (size_t scanlineIdx = 0; scanlineIdx < numScanlines; scanlineIdx++)
				{
					// the angle of the scanline
					double scanlineAngle = -fov / 2 + scanlineIdx / (numScanlines - 1) * fov;
					m_txParameters[scanlineIdx] = getTxScanline3D(activeAperture, txAperture, vec2{ scanlineStartX, 0 }, vec2{ scanlineAngle, 0 });
				}
			}
			else if (m_type == Biphased)
			{
				bool fullApertureX = m_maxApertureSize.x == elementLayout.x;
				bool fullApertureY = m_maxApertureSize.y == elementLayout.y;

				auto elementCenterpoints = m_pTransducer->getElementCenterPoints();
				size_t scanlinesDone = 0;
				for (size_t scanlineIdxY = 0; scanlineIdxY < m_numScanlines.y; scanlineIdxY++)
				{
					for (size_t scanlineIdxX = 0; scanlineIdxX < m_numScanlines.x; scanlineIdxX++)
					{
						//vec2s firstElementIdx;
						//vec2s lastElementIdx;
						vec2 scanlineStart;

						// the scanline position in terms of elementIndices
						vec2 scanlinePositionRelative;
						scanlinePositionRelative =
							static_cast<vec2>(vec2s{ scanlineIdxX, scanlineIdxY }) /
							static_cast<vec2>(m_numScanlines - 1) *
							static_cast<vec2>(elementLayout - m_maxApertureSize)
							+ (m_maxApertureSize - 1) / 2;

						rect2s activeAperture = computeAperture(elementLayout, m_maxApertureSize, scanlinePositionRelative);
						rect2s txAperture = computeAperture(elementLayout, m_txMaxApertureSize, scanlinePositionRelative);

						scanlineStart.x = (elementCenterpoints->at(activeAperture.begin.x).x +
							elementCenterpoints->at(activeAperture.end.x).x) / 2;
						scanlineStart.y = (elementCenterpoints->at(activeAperture.begin.y *elementLayout.x).y +
							elementCenterpoints->at(activeAperture.end.y   *elementLayout.x).y) / 2;

						// the angle of the scanline
						vec2 scanlineAngle = (vec2{ (double)scanlineIdxX, (double)scanlineIdxY } / (m_numScanlines - 1) - 0.5) * m_fov;
						m_txParameters[scanlinesDone] = getTxScanline3D(activeAperture, txAperture, scanlineStart, scanlineAngle);
						scanlinesDone++;
					}
				}
			}
			else if (m_type == Planewave)
			{
				if (m_pTransducer->getType() != USTransducer::Linear)
				{
					// only linear supported for now
					logging::log_error("Beamformer: Planewave only supported for linear arrays, yet.");
					throw std::invalid_argument("Imaging type not implemented yet");
				}

				auto elementCenterpoints = m_pTransducer->getElementCenterPoints();
				vec firstElement = elementCenterpoints->at(0);
				vec lastElement = elementCenterpoints->at(m_pTransducer->getNumElements() - 1);

				double angleRad = m_txSteeringAngle.x * M_PI / 180.0;
				
				// the position of the scanline on the x axis is by default at center for planwave imaging
				double scanlinePosition = firstElement.x +
					0.5 * (lastElement.x - firstElement.x);
				
				// the scanline position in terms of elementIndices at center of array
				double scanlinePositionRelative = 0.5 * (m_pTransducer->getNumElements() - 1);

				rect2s activeAperture = computeAperture(elementLayout, m_maxApertureSize, { scanlinePositionRelative, 0 });
				rect2s txAperture = computeAperture(elementLayout, m_txMaxApertureSize, { scanlinePositionRelative, 0 });

				m_txParameters[0] = getTxScanline3D(activeAperture, txAperture, vec2d{ scanlinePosition, 0 }, vec2d{ angleRad, 0 });
			}
			else {
				logging::log_error("Beamformer: Imaging type not implemented yet");
				throw std::invalid_argument("Imaging type not implemented yet");
			}

			/*for (size_t scanlineIdxY = 0; scanlineIdxY < m_numScanlines.y; scanlineIdxY++)
			{
				for (size_t scanlineIdxX = 0; scanlineIdxX < m_numScanlines.x; scanlineIdxX++)
				{
					size_t scanlineIdx = scanlineIdxX + scanlineIdxY * m_numScanlines.x;
					log_always("computeTxParameters: idx (", scanlineIdxX, ", ", scanlineIdxY, "), pos: (",
						m_txParameters[scanlineIdx].position.x, ", ",
						m_txParameters[scanlineIdx].position.y, ", ",
						m_txParameters[scanlineIdx].position.z, "), dir (",
						m_txParameters[scanlineIdx].direction.x, ", ",
						m_txParameters[scanlineIdx].direction.y, ", ",
						m_txParameters[scanlineIdx].direction.z, " firstActive (",
						m_txParameters[scanlineIdx].firstActiveElementIndex.x, ", ",
						m_txParameters[scanlineIdx].firstActiveElementIndex.y, ") lastactive (",
						m_txParameters[scanlineIdx].lastActiveElementIndex.x, ", ",
						m_txParameters[scanlineIdx].lastActiveElementIndex.y, ")");
				}
			}*/

			//Compute the RX scanlines based on the TX scanlines, but include subdivision
			//Subdivide the tx scanlines
			m_rxParameters = make_shared<std::vector<std::vector<ScanlineRxParameters3D> > >();
			m_rxParameters->resize(m_numRxScanlines.x, vector<ScanlineRxParameters3D>(m_numRxScanlines.y));
			if (m_numScanlines.y > 1)
			{
				for (size_t scanlineIdxY = 0; scanlineIdxY < (m_numScanlines.y - 1); scanlineIdxY++)
				{
					for (size_t scanlineIdxX = 0; scanlineIdxX < (m_numScanlines.x - 1); scanlineIdxX++)
					{
						//subdivide between
						//  (scanlineIdxX,   scanlineIdxY),
						//  (scanlineIdxX+1, scanlineIdxY),
						//  (scanlineIdxX,   scanlineIdxY+1),
						//  (scanlineIdxX+1, scanlineIdxY+1)
						size_t txScanlineIdx1 = scanlineIdxX + scanlineIdxY      * m_numScanlines.x;
						size_t txScanlineIdx2 = (scanlineIdxX + 1) + scanlineIdxY      * m_numScanlines.x;
						size_t txScanlineIdx3 = scanlineIdxX + (scanlineIdxY + 1) * m_numScanlines.x;
						size_t txScanlineIdx4 = (scanlineIdxX + 1) + (scanlineIdxY + 1) * m_numScanlines.x;
						auto txScanline1 = m_txParameters[txScanlineIdx1];
						auto txScanline2 = m_txParameters[txScanlineIdx2];
						auto txScanline3 = m_txParameters[txScanlineIdx3];
						auto txScanline4 = m_txParameters[txScanlineIdx4];
						for (size_t rxScanIdxY = 0; rxScanIdxY < m_rxScanlineSubdivision.y; rxScanIdxY++)
						{
							for (size_t rxScanIdxX = 0; rxScanIdxX < m_rxScanlineSubdivision.x; rxScanIdxX++)
							{
								vec2 interp = vec2{ static_cast<double>(rxScanIdxX), static_cast<double>(rxScanIdxY) }
								/ static_cast<vec2>(m_rxScanlineSubdivision);

								//interpolate...
								ScanlineRxParameters3D interpolated =
									getRxScanline3DInterpolated(
										txScanlineIdx1, txScanline1,
										txScanlineIdx2, txScanline2,
										txScanlineIdx3, txScanline3,
										txScanlineIdx4, txScanline4,
										interp);
								//and store
								(*m_rxParameters)[scanlineIdxX*m_rxScanlineSubdivision.x + rxScanIdxX][scanlineIdxY*m_rxScanlineSubdivision.y + rxScanIdxY] =
									interpolated;
							}
						}
					}
					// now only interpolate within the last Y-plane (between the current two X-Planes)
					size_t scanlineIdxX = (m_numScanlines.x - 1);
					//subdivide between
					//  (scanlineIdxX,   scanlineIdxY),
					//  (scanlineIdxX,   scanlineIdxY),
					//  (scanlineIdxX,   scanlineIdxY+1),
					//  (scanlineIdxX,   scanlineIdxY+1)
					size_t txScanlineIdx1 = scanlineIdxX + scanlineIdxY      * m_numScanlines.x;
					size_t txScanlineIdx2 = scanlineIdxX + scanlineIdxY      * m_numScanlines.x;
					size_t txScanlineIdx3 = scanlineIdxX + (scanlineIdxY + 1) * m_numScanlines.x;
					size_t txScanlineIdx4 = scanlineIdxX + (scanlineIdxY + 1) * m_numScanlines.x;
					auto txScanline1 = m_txParameters[txScanlineIdx1];
					auto txScanline2 = m_txParameters[txScanlineIdx2];
					auto txScanline3 = m_txParameters[txScanlineIdx3];
					auto txScanline4 = m_txParameters[txScanlineIdx4];
					size_t rxScanIdxX = 0;
					for (size_t rxScanIdxY = 0; rxScanIdxY < m_rxScanlineSubdivision.y; rxScanIdxY++)
					{
						vec2 interp = vec2{ static_cast<double>(rxScanIdxX), static_cast<double>(rxScanIdxY) }
						/ static_cast<vec2>(m_rxScanlineSubdivision);

						//interpolate...
						ScanlineRxParameters3D interpolated =
							getRxScanline3DInterpolated(
								txScanlineIdx1, txScanline1,
								txScanlineIdx2, txScanline2,
								txScanlineIdx3, txScanline3,
								txScanlineIdx4, txScanline4,
								interp);
						//and store
						(*m_rxParameters)[scanlineIdxX*m_rxScanlineSubdivision.x + rxScanIdxX][scanlineIdxY*m_rxScanlineSubdivision.y + rxScanIdxY] =
							interpolated;
					}
					//Add the last scanline in x from this plane
					size_t txScanlineIdx = (m_numScanlines.x - 1) + scanlineIdxY * m_numScanlines.x;
					(*m_rxParameters)[(m_numScanlines.x - 1) * m_rxScanlineSubdivision.x][scanlineIdxY * m_rxScanlineSubdivision.y] =
						getRxScanline3D(txScanlineIdx, m_txParameters[txScanlineIdx]);
				}
				// now only interpolate within the last X-plane
				size_t scanlineIdxY = m_numScanlines.y - 1;
				for (size_t scanlineIdxX = 0; scanlineIdxX < (m_numScanlines.x - 1); scanlineIdxX++)
				{
					//subdivide between
					//  (scanlineIdxX,   scanlineIdxY),
					//  (scanlineIdxX+1, scanlineIdxY),
					//  (scanlineIdxX,   scanlineIdxY),
					//  (scanlineIdxX+1, scanlineIdxY)
					size_t txScanlineIdx1 = scanlineIdxX + scanlineIdxY * m_numScanlines.x;
					size_t txScanlineIdx2 = (scanlineIdxX + 1) + scanlineIdxY * m_numScanlines.x;
					size_t txScanlineIdx3 = scanlineIdxX + scanlineIdxY * m_numScanlines.x;
					size_t txScanlineIdx4 = (scanlineIdxX + 1) + scanlineIdxY * m_numScanlines.x;
					auto txScanline1 = m_txParameters[txScanlineIdx1];
					auto txScanline2 = m_txParameters[txScanlineIdx2];
					auto txScanline3 = m_txParameters[txScanlineIdx3];
					auto txScanline4 = m_txParameters[txScanlineIdx4];
					size_t rxScanIdxY = 0;
					for (size_t rxScanIdxX = 0; rxScanIdxX < m_rxScanlineSubdivision.x; rxScanIdxX++)
					{
						vec2 interp = vec2{ static_cast<double>(rxScanIdxX), static_cast<double>(rxScanIdxY) }
						/ static_cast<vec2>(m_rxScanlineSubdivision);

						//interpolate...
						ScanlineRxParameters3D interpolated =
							getRxScanline3DInterpolated(
								txScanlineIdx1, txScanline1,
								txScanlineIdx2, txScanline2,
								txScanlineIdx3, txScanline3,
								txScanlineIdx4, txScanline4,
								interp);
						//and store
						(*m_rxParameters)[scanlineIdxX*m_rxScanlineSubdivision.x + rxScanIdxX][scanlineIdxY*m_rxScanlineSubdivision.y + rxScanIdxY] =
							interpolated;
					}
				}
				//add the very last scanline (in x and y)
				size_t txScanlineIdx = (m_numScanlines.x - 1) + (m_numScanlines.y - 1) * m_numScanlines.x;
				(*m_rxParameters)[(m_numScanlines.x - 1)*m_rxScanlineSubdivision.x][(m_numScanlines.y - 1)*m_rxScanlineSubdivision.y] =
					getRxScanline3D(txScanlineIdx, m_txParameters[txScanlineIdx]);
			}
			else if (m_numScanlines.x > 1) 
			{
				// Normal 2D planar scanline imaging
				for (size_t scanlineIdxX = 0; scanlineIdxX < (m_numScanlines.x - 1); scanlineIdxX++)
				{
					//subdivide between
					//  (scanlineIdxX,   0),
					//  (scanlineIdxX+1, 0),

					size_t txScanlineIdx1 = scanlineIdxX;
					size_t txScanlineIdx2 = (scanlineIdxX + 1);
					size_t txScanlineIdx3 = scanlineIdxX;
					size_t txScanlineIdx4 = (scanlineIdxX + 1);
					auto txScanline1 = m_txParameters[txScanlineIdx1];
					auto txScanline2 = m_txParameters[txScanlineIdx2];
					auto txScanline3 = m_txParameters[txScanlineIdx3];
					auto txScanline4 = m_txParameters[txScanlineIdx4];
					for (size_t rxScanIdxX = 0; rxScanIdxX < m_rxScanlineSubdivision.x; rxScanIdxX++)
					{
						vec2 interp = vec2{ static_cast<double>(rxScanIdxX), static_cast<double>(0) }
						/ static_cast<vec2>(m_rxScanlineSubdivision);

						//interpolate...
						ScanlineRxParameters3D interpolated =
							getRxScanline3DInterpolated(
								txScanlineIdx1, txScanline1,
								txScanlineIdx2, txScanline2,
								txScanlineIdx3, txScanline3,
								txScanlineIdx4, txScanline4,
								interp);
						//and store
						(*m_rxParameters)[rxScanIdxX + scanlineIdxX*m_rxScanlineSubdivision.x][0] = interpolated;
					}
				}
				(*m_rxParameters)[(m_numScanlines.x - 1)*m_rxScanlineSubdivision.x][0] = getRxScanline3D((m_numScanlines.x - 1), m_txParameters[(m_rxScanlineSubdivision.x - 1)]);
			}
			else
			{
				// Plane wave imaging

				if (m_pTransducer->getType() != USTransducer::Linear)
				{
					logging::log_error("Beamformer: Planewave imaging supported for linear probes only.");
					throw std::invalid_argument("Imaging type not implemented yet");
				}

				// based on setup tx-rx scanline paradigm, we will create artificial tx scanlines for each rx scanline and simply take it
				size_t numScanlines = m_numRxScanlines.x;
				auto elementCenterpoints = m_pTransducer->getElementCenterPoints();
				vec firstElement = elementCenterpoints->at(0);
				vec lastElement = elementCenterpoints->at(m_pTransducer->getNumElements() - 1);
				double angleRad = m_txSteeringAngle.x * M_PI / 180.0;
				
				//evenly space the receive scanlines between the first and last element
				for (size_t rxScanlineIdx = 0; rxScanlineIdx < numScanlines; rxScanlineIdx++)
				{
					// the position of the scanline on the x axis
					double scanlinePosition = firstElement.x +
						static_cast<double>(rxScanlineIdx) / (numScanlines - 1) * (lastElement.x - firstElement.x);
					
					// the scanline position in terms of elementIndices
					double scanlinePositionRelative = static_cast<double>(rxScanlineIdx) / (numScanlines - 1) * (m_pTransducer->getNumElements() - 1);

					rect2s activeAperture = computeAperture(elementLayout, m_maxApertureSize, { scanlinePositionRelative, 0 });
					rect2s txAperture = computeAperture(elementLayout, m_txMaxApertureSize, { scanlinePositionRelative, 0 });

					ScanlineTxParameters3D tmpTxParameters = getTxScanline3D(activeAperture, txAperture, vec2d{ scanlinePosition, 0 }, vec2d{ angleRad, 0 });
				
					// get copy of rx scanline
					ScanlineRxParameters3D currentRxLine = getRxScanline3D(0, tmpTxParameters);
				
					//and store
					(*m_rxParameters)[rxScanlineIdx][0] = currentRxLine;
				}
		 	}

			logging::log_info("Beamformer: computing TX parameters finished");
		}
		else
		{
			logging::log_always("Beamformer: Imaging parameters are not fully defined, yet.");
		}
	}

	const std::vector<ScanlineTxParameters3D>* Beamformer::getTxParameters()
	{
		return &m_txParameters;
	}

	std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > Beamformer::getRxParameters()
	{
		return m_rxParameters;
	}

	shared_ptr<const RxBeamformerCuda> Beamformer::getCurrentRxBeamformer()
	{
		size_t numDepths = m_numSamplesRecon;

		//TODO if we get new parameters, create new RxBeamformer if neccessary
		if (!m_pRxBeamformer)
		{
			m_pRxBeamformer = make_shared<RxBeamformerCuda>(
				m_rxParameters, numDepths, m_depth,
				m_speedOfSoundMMperS, m_pTransducer);
		}
		return m_pRxBeamformer;
	}

	ScanlineTxParameters3D Beamformer::getTxScanline3D(
		rect2s activeAperture,
		rect2s txAperture,
		vec2d scanlineStart,
		vec2d steeringAngle)
	{
		auto elementCenterpoints = m_pTransducer->getElementCenterPoints();
		vec2s elementLayout = m_pTransducer->getElementLayout();
		vec scanlineStart3 = vec{ scanlineStart.x, scanlineStart.y, 0 };

		//store the three elements in the corners and their relative indices (for focus calculation)
		// <0>: relativeIndex (0 to 1, double)
		// <1>: elementIndex (0 to physical number of elements)
		// <2>: elementPos (mm physical) 
		vector<tuple<vec2, vec2s, vec> > cornerElements;
		bool is3D = false;
		if (txAperture.begin.x == txAperture.end.x && txAperture.begin.y == txAperture.end.y)
		{
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({0.5, 0.5}, txAperture.begin, elementCenterpoints->at(txAperture.begin.x + txAperture.begin.y*elementLayout.x)));
		}
		else if (txAperture.begin.x == txAperture.end.x)
		{
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({0.5,   0}, txAperture.begin, elementCenterpoints->at(txAperture.begin.x + txAperture.begin.y*elementLayout.x)));
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({0.5,   1}, txAperture.end, elementCenterpoints->at(txAperture.end.x + txAperture.end.y*elementLayout.x)));
		}
		else if (txAperture.begin.y == txAperture.end.y)
		{
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({  0, 0.5}, txAperture.begin, elementCenterpoints->at(txAperture.begin.x + txAperture.begin.y*elementLayout.x)));
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({  1, 0.5}, txAperture.end, elementCenterpoints->at(txAperture.end.x + txAperture.end.y*elementLayout.x)));
		}
		else
		{
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({  0,   0}, { txAperture.begin.x, txAperture.begin.y }, elementCenterpoints->at(txAperture.begin.x + txAperture.begin.y*elementLayout.x)));
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({  0,   1}, { txAperture.begin.x, txAperture.end.y   }, elementCenterpoints->at(txAperture.begin.x + txAperture.end.y*elementLayout.x)));
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({  1,   0}, { txAperture.end.x,   txAperture.begin.y }, elementCenterpoints->at(txAperture.end.x + txAperture.begin.y*elementLayout.x)));
			cornerElements.push_back(std::tuple<vec2, vec2s, vec>({  1,   1}, { txAperture.end.x,   txAperture.end.y   }, elementCenterpoints->at(txAperture.end.x + txAperture.end.y*elementLayout.x)));
			is3D = true;
		}

		ScanlineTxParameters3D params;
		params.firstActiveElementIndex = activeAperture.begin;
		params.lastActiveElementIndex = activeAperture.end;
		params.txAperture = txAperture;

		// next, set the correct elements according to defined aperture (general and transmit) in the scanline parameters 
		// fill the elementMap ScanlineTxParameters
		params.elementMap.resize(elementLayout.x, vector<bool>(elementLayout.y, false)); //set all to false
		for (size_t activeElementIdxX = activeAperture.begin.x; activeElementIdxX <= activeAperture.end.x; activeElementIdxX++)
		{
			for (size_t activeElementIdxY = activeAperture.begin.y; activeElementIdxY <= activeAperture.end.y; activeElementIdxY++)
			{
				params.elementMap[activeElementIdxX][activeElementIdxY] = true;
			}
		}
		// fill the transmit elementMap ScanlineTxParameters
		params.txElementMap.resize(elementLayout.x, vector<bool>(elementLayout.y, false)); //set all to false
		for (size_t activeElementIdxX = txAperture.begin.x; activeElementIdxX <= txAperture.end.x; activeElementIdxX++)
		{
			for (size_t activeElementIdxY = txAperture.begin.y; activeElementIdxY <= txAperture.end.y; activeElementIdxY++)
			{
				params.txElementMap[activeElementIdxX][activeElementIdxY] = true;
			}
		}

		// calculate the scanline local coordinate system such that scanlines are pointing in z-direction
		// R = [scanlinePerpDirX', scanlinePerpDirY', scanlineDir]
		// where the X and Y are the parts of [1,0,0] and [0,1,0] perpendicular to scanlineDir
		// and X being lateral, Y being elevational
		vec scanlineDir = normalize(vec{ tan(steeringAngle.x), tan(steeringAngle.y), 1 });
		vec scanlinePerpDirX = normalize(vec{ 1,0,0 } -scanlineDir*dot(vec{ 1,0,0 }, scanlineDir));
		vec scanlinePerpDirY = normalize(vec{ 0,1,0 } -scanlineDir*dot(vec{ 0,1,0 }, scanlineDir));
		// sanity check that constructed transform R is valid and righthanded
		assert(abs(dot(scanlineDir, scanlinePerpDirX)) < 1e-7);
		assert(abs(dot(scanlineDir, scanlinePerpDirY)) < 1e-7);
		assert(abs(dot(scanlinePerpDirX, scanlinePerpDirY)) < 1e-7);

		//compute the delays
		params.delays.resize(txAperture.end.x - txAperture.begin.x + 1, vector<double>(txAperture.end.y - txAperture.begin.y + 1, 0));
		double maxDelay = 0.0;
		if (m_txFocusActive)
		{
			// with focus
			double maxTransitTime = 0;
			for (auto t : cornerElements)
			{
				auto relativeIndex = std::get<0>(t);
				auto elementIndex = std::get<1>(t);
				auto elementPos = std::get<2>(t);

				vec focusPointFromFocusCenter =
					((relativeIndex.x - 0.5)*m_txFocusWidth)*scanlinePerpDirX +
					((relativeIndex.y - 0.5)*m_txFocusWidth)*scanlinePerpDirY;
				vec elementToFocus = scanlineStart3 + m_txFocusDepth*scanlineDir + focusPointFromFocusCenter - elementPos;
				double transitTime = m_pTransducer->computeTransitTime(elementIndex, elementToFocus, m_speedOfSoundMMperS, m_correctMatchingLayers);
				maxTransitTime = max(maxTransitTime, transitTime);
			}

			for (size_t activeElementIdxX = txAperture.begin.x; activeElementIdxX <= txAperture.end.x; activeElementIdxX++)
			{
				for (size_t activeElementIdxY = txAperture.begin.y; activeElementIdxY <= txAperture.end.y; activeElementIdxY++)
				{
					size_t localElementIdxX = activeElementIdxX - txAperture.begin.x;
					size_t localElementIdxY = activeElementIdxY - txAperture.begin.y;
					vec2 relativeIndex = vec2{ static_cast<double>(localElementIdxX), static_cast<double>(localElementIdxY) }
					/ static_cast<vec2>(supra::max(txAperture.end - txAperture.begin, { 1, 1 }));
					if (txAperture.end.x == txAperture.begin.x)
					{
						relativeIndex.x = 0.5;
					}
					if (txAperture.end.y == txAperture.begin.y)
					{
						relativeIndex.y = 0.5;
					}

					vec elementCenter = elementCenterpoints->at(activeElementIdxX + activeElementIdxY*elementLayout.x);

					vec focusPointFromFocusCenter =
						((relativeIndex.x - 0.5)*m_txFocusWidth)*scanlinePerpDirX +
						((relativeIndex.y - 0.5)*m_txFocusWidth)*scanlinePerpDirY;
					vec elementToFocus = scanlineStart3 + m_txFocusDepth*scanlineDir + focusPointFromFocusCenter - elementCenter;
					double transitTime = m_pTransducer->computeTransitTime(vec2s{ activeElementIdxX, activeElementIdxY }, elementToFocus, m_speedOfSoundMMperS, m_correctMatchingLayers);

					double delay = maxTransitTime - transitTime;
					params.delays[localElementIdxX][localElementIdxY] = delay;
					maxDelay = max(maxDelay, delay);
				}
			}
		}
		else {
			// without focus
			double maxTransitTime = 0;
			for (auto t : cornerElements)
			{
				auto elementPos = std::get<2>(t);
				vec d = (scanlineStart3 - elementPos)*scanlineDir;
				maxTransitTime = max(maxTransitTime, (d.x + d.y + d.z + m_txFocusDepth) / m_speedOfSoundMMperS);
			}

			for (size_t activeElementIdxX = txAperture.begin.x; activeElementIdxX <= txAperture.end.x; activeElementIdxX++)
			{
				for (size_t activeElementIdxY = txAperture.begin.y; activeElementIdxY <= txAperture.end.y; activeElementIdxY++)
				{
					size_t localElementIdxX = activeElementIdxX - txAperture.begin.x;
					size_t localElementIdxY = activeElementIdxY - txAperture.begin.y;
					vec elementCenter = elementCenterpoints->at(activeElementIdxX + activeElementIdxY*elementLayout.x);

					//TODO check this. does not seem right
					vec d = (scanlineStart3 - elementCenter)*scanlineDir;
					double transitTime = (d.x + d.y + d.z + m_txFocusDepth) / m_speedOfSoundMMperS;

					double delay = maxTransitTime - transitTime;
					params.delays[localElementIdxX][localElementIdxY] = delay;
					maxDelay = max(maxDelay, delay);
				}
			}
		}
		params.maxDelay = maxDelay;

		//compute the weights
		// for that first compute the maximum distance of tx elements to the scanline, it serves as normalization for the window position
		vec2 maxElementScanlineDist = { 0, 0 };
		for (auto element : cornerElements)
		{
			auto elementPos = std::get<2>(element);
			maxElementScanlineDist = max(maxElementScanlineDist,
				vec2{ static_cast<double>(std::abs(elementPos.x - scanlineStart.x)),
					  static_cast<double>(std::abs(elementPos.y - scanlineStart.y)) });
		}

		WindowFunction win(m_txWindow, m_txWindowParameter);
		params.weights.resize(txAperture.end.x - txAperture.begin.x + 1, vector<double>(txAperture.end.y - txAperture.begin.y + 1, 0));
		for (size_t activeElementIdxX = txAperture.begin.x; activeElementIdxX <= txAperture.end.x; activeElementIdxX++)
		{
			for (size_t activeElementIdxY = txAperture.begin.y; activeElementIdxY <= txAperture.end.y; activeElementIdxY++)
			{
				size_t localElementIdxX = activeElementIdxX - txAperture.begin.x;
				size_t localElementIdxY = activeElementIdxY - txAperture.begin.y;

				vec elementCenter = elementCenterpoints->at(activeElementIdxX + activeElementIdxY*elementLayout.x);
				vec2 elementScanlineDistance = { elementCenter.x - scanlineStart.x, elementCenter.z - scanlineStart.y };
				vec2 elementRelativeDistance = elementScanlineDistance / maxElementScanlineDist;

				if (txAperture.begin.x == txAperture.end.x)
				{
					elementRelativeDistance.x = 0.0f;
				}
				if (txAperture.begin.y == txAperture.end.y)
				{
					elementRelativeDistance.y = 0.0f;
				}

				double weight;
				if(is3D)
				{
					weight = std::sqrt(win.get(static_cast<float>(elementRelativeDistance.x))*win.get(static_cast<float>(elementRelativeDistance.y)));
				}
				else
				{
					weight = win.get(static_cast<float>(elementRelativeDistance.x))*win.get(static_cast<float>(elementRelativeDistance.y));
				}
				params.weights[localElementIdxX][localElementIdxY] = weight;
			}
		}
		switch(m_txWindow)
		{
		case WindowRectangular:
		case WindowINVALID:
		default:
			// for now: rectangular!
			params.weights.resize(txAperture.end.x - txAperture.begin.x + 1, vector<double>(txAperture.end.y - txAperture.begin.y + 1, 0));
			break;

		}

		params.position = vec{ scanlineStart.x, 0, scanlineStart.y };
		params.direction = vec{ scanlineDir.x, scanlineDir.z, scanlineDir.y };

		return params;
	}

	ScanlineRxParameters3D Beamformer::getRxScanline3DInterpolated(
		size_t txScanlineIdx1, const ScanlineTxParameters3D& txScanline1,
		size_t txScanlineIdx2, const ScanlineTxParameters3D& txScanline2,
		size_t txScanlineIdx3, const ScanlineTxParameters3D& txScanline3,
		size_t txScanlineIdx4, const ScanlineTxParameters3D& txScanline4,
		vec2 interp)
	{
		assert(txScanlineIdx1 < std::numeric_limits<uint16_t>::max());
		assert(txScanlineIdx2 < std::numeric_limits<uint16_t>::max());
		assert(txScanlineIdx3 < std::numeric_limits<uint16_t>::max());
		assert(txScanlineIdx4 < std::numeric_limits<uint16_t>::max());

		auto elementCenterpoints = m_pTransducer->getElementCenterPoints();
		vec2s elementLayout = m_pTransducer->getElementLayout();

		ScanlineRxParameters3D rx;
		//decide which tx scanline it will be based on
		rx.txParameters[0] = txScanline1;
		rx.txParameters[1] = txScanline2;
		rx.txParameters[2] = txScanline3;
		rx.txParameters[3] = txScanline4;
		rx.txParameters[0].txScanlineIdx = static_cast<uint16_t>(txScanlineIdx1);
		rx.txParameters[1].txScanlineIdx = static_cast<uint16_t>(txScanlineIdx2);
		rx.txParameters[2].txScanlineIdx = static_cast<uint16_t>(txScanlineIdx3);
		rx.txParameters[3].txScanlineIdx = static_cast<uint16_t>(txScanlineIdx4);

		rx.txWeights[0] = (1 - interp.x) * (1 - interp.y);
		rx.txWeights[1] = interp.x  * (1 - interp.y);
		rx.txWeights[2] = (1 - interp.x) *      interp.y;
		rx.txWeights[3] = interp.x  *      interp.y;

		//fix position and direction
		vec position = (1 - interp.y)*((1 - interp.x)*txScanline1.position +
			interp.x *txScanline2.position) +
			interp.y *((1 - interp.x)*txScanline3.position +
				interp.x *txScanline4.position);
		vec direction =
			slerp3(
				slerp3(txScanline1.direction, txScanline2.direction, interp.x),
				slerp3(txScanline3.direction, txScanline4.direction, interp.x),
				interp.y);
		rx.position = position;
		rx.direction = direction;

		//determine maximum distances from the scanline, they serve as normalization
		vector<vec> cornerElements;

		for (auto txParam : rx.txParameters)
		{
			cornerElements.push_back(elementCenterpoints->at(txParam.firstActiveElementIndex.x + txParam.firstActiveElementIndex.y*elementLayout.x));
			cornerElements.push_back(elementCenterpoints->at(txParam.firstActiveElementIndex.x + txParam.lastActiveElementIndex.y*elementLayout.x));
			cornerElements.push_back(elementCenterpoints->at(txParam.lastActiveElementIndex.x + txParam.firstActiveElementIndex.y*elementLayout.x));
			cornerElements.push_back(elementCenterpoints->at(txParam.lastActiveElementIndex.x + txParam.lastActiveElementIndex.y*elementLayout.x));
		}
		vec2 maxDist = { 0, 0 };
		for (auto elementPos : cornerElements)
		{
			maxDist = max(maxDist,
				vec2{ static_cast<double>(std::abs(elementPos.x - rx.position.x)),
					  static_cast<double>(std::abs(elementPos.y - rx.position.z)) });
		}
		rx.maxElementDistance = maxDist;

		//compute scanline scaling from scanline angles
		//vec2 scanlineAngle = vec2{atan(rx.direction.x / rx.direction.y), atan(rx.direction.z / rx.direction.y)};

		return rx;
	}

	ScanlineRxParameters3D Beamformer::getRxScanline3D(size_t txScanlineIdx, const ScanlineTxParameters3D& txScanline)
	{
		return getRxScanline3DInterpolated(
			txScanlineIdx, txScanline,
			txScanlineIdx, txScanline,
			txScanlineIdx, txScanline,
			txScanlineIdx, txScanline,
			{ 0, 0 });
	}

	rect2s Beamformer::computeAperture(vec2s layout, vec2s apertureSize, vec2 relativePosition)
	{
		assert(apertureSize.x <= layout.x && apertureSize.y <= layout.y);

		//find the closest elements -> they have to be active
		// first by pure geometry
		vec2 apertureBegin = relativePosition - static_cast<vec2>(apertureSize - 1) / 2;
		vec2 apertureEnd = relativePosition + static_cast<vec2>(apertureSize - 1) / 2;

		//and then fix them iff necessary
		// X
		if (round(apertureBegin.x) < 0)
		{
			apertureBegin.x = 0;
			apertureEnd.x = static_cast<double>(apertureSize.x - 1);
		}
		else if (round(apertureEnd.x) > (layout.x - 1))
		{
			apertureEnd.x = static_cast<double>(layout.x - 1);
			apertureBegin.x = apertureEnd.x - (apertureSize.x - 1);
		}
		// Y
		if (round(apertureBegin.y) < 0)
		{
			apertureBegin.y = 0;
			apertureEnd.y = static_cast<double>(apertureSize.y - 1);
		}
		else if (round(apertureEnd.y) > (layout.y - 1))
		{
			apertureEnd.y = static_cast<double>(layout.y - 1);
			apertureBegin.y = apertureEnd.y - (apertureSize.y - 1);
		}

		rect2s aperture;
		aperture.begin = static_cast<vec2s>(round(apertureBegin));
		aperture.end = static_cast<vec2s>(round(apertureEnd));

		return aperture;
	}
}
