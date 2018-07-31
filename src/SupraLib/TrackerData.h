// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2011-2016, all rights reserved,
//      Christoph Hennersperger 
//		EmaiL christoph.hennersperger@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
//	and
//		Rüdiger Göbl
//		Email r.goebl@tum.de
//
// ================================================================================================

#ifndef __TRACKERDATA_H__
#define __TRACKERDATA_H__

#include "RecordObject.h"
#include <cstdint>
#include <vector>
#include <array>
#include <string>

namespace supra
{
	/// A pose as measured by a tracking system
	class TrackerData
	{
	public:
		/// Type for a matrix
		typedef std::array<double, 16> Matrix;

		/// Base constructor
		TrackerData();
		/// Constructs TrackerData from a rigid transform matrix in a std::array of length 16
		TrackerData(const std::array<double, 16>& matrix, int32_t qual, int32_t uid, std::string instrumentName, double time);
		/// Constructs TrackerData from a rigid transform matrix in a std::vector of length 16
		TrackerData(const std::vector<double>& matrix, int32_t qual, int32_t uid, std::string instrumentName, double time);
		/// Constructs TrackerData from a position and quaternion, passed as arrays of length 3 and 4
		TrackerData(const double pos[3], const double rot[4], int32_t qual, int32_t uid, std::string instrumentName, double time);
		/// Constructs TrackerData from a position and quaternion, passed as std::vectors of length 3 and 4
		TrackerData(const std::vector<double>& pos, const std::vector<double>& quat, int32_t qual, int32_t uid, std::string instrumentName, double time);
		/// Constructs TrackerData from a position and quaternion, passed as scalars
		TrackerData(const double posX, const double posY, const double posZ,
			const double rotQ1, const double rotQ2, const double rotQ3, const double rotQ4,
			const int32_t qual, const int32_t uind, std::string instrumentName, const double time);
		/// Copy constructor
		TrackerData(const TrackerData& value);

		/// Assignment operator
		TrackerData& operator=(const TrackerData& a);

		/// Returns the quality of the tracker measurement
		inline const int getQuality() const { return m_quality; };

		/// Returns the identifier of the pose within the sequence recorded by the tracker
		inline const int getUID() const { return m_uid; };
		/// Returns the name of the tracked instrument
		inline const std::string& getInstrumentName() const { return m_instrumentName; };

		/// returns the pose in a row major transformation matrix
		inline const std::array<double, 16>& getMatrix() const {
			return m_matrix;
		}

		/// Prints the pose matrix
		void print();

	private:
		void setPositionQuaternion(const std::vector<double>& pos, const std::vector<double>& quat);

		std::array<double, 16> m_matrix;	// Rigid transformation matrix

		int32_t m_uid;						// tracker identifier
		std::string m_instrumentName;		// tracker identifier
		int32_t m_quality;					// quality of data
	};
}

#endif //!__TRACKERDATA_H__