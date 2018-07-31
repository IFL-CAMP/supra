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

#include "TrackerData.h"

#include <vector>
#include <iostream>
#include <assert.h>
#include <utilities/utility.h>
#include <utilities/Logging.h>

namespace supra
{
	TrackerData::TrackerData()
		: m_quality(100000000), m_uid(0), m_instrumentName("")

	{
		m_matrix.fill(0);
		m_matrix[0] = 1;
		m_matrix[5] = 1;
		m_matrix[10] = 1;
		m_matrix[15] = 1;
	}

	TrackerData::TrackerData(const std::array<double, 16>& matrix, int32_t qual, int32_t uid, std::string instrumentName, double time)
		: m_quality(qual), m_uid(uid), m_instrumentName(instrumentName)
	{
		m_matrix = matrix;
	}

	TrackerData::TrackerData(const std::vector<double>& matrix, int32_t qual, int32_t uid, std::string instrumentName, double time)
		: m_quality(qual), m_uid(uid), m_instrumentName(instrumentName)
	{
		if (matrix.size() != 16)
			return;

		std::copy_n(matrix.begin(), 16, m_matrix.begin());
	}

	TrackerData::TrackerData(const double pos[3], const double rot[4], int32_t qual, int32_t uid, std::string instrumentName, double time)
		: m_quality(qual), m_uid(uid), m_instrumentName(instrumentName)
	{
		setPositionQuaternion({ pos[0], pos[1], pos[2] }, { rot[0], rot[1], rot[2], rot[3] });
	}

	TrackerData::TrackerData(const std::vector<double>& pos, const std::vector<double>& quat, int32_t qual, int32_t uid, std::string instrumentName, double time)
		: m_quality(qual), m_uid(uid), m_instrumentName(instrumentName)
	{
		if (pos.size() != 3 || quat.size() != 4)
			return;

		setPositionQuaternion(pos, quat);
	}

	TrackerData::TrackerData(const double posX, const double posY, const double posZ,
		const double rotQ1, const double rotQ2, const double rotQ3, const double rotQ4,
		const int32_t qual, const int32_t uid, std::string instrumentName, const double time)
		: m_quality(qual), m_uid(uid), m_instrumentName(instrumentName)
	{
		setPositionQuaternion({ posX, posY, posZ }, { rotQ1, rotQ2, rotQ3, rotQ4 });
	}


	TrackerData::TrackerData(const TrackerData& a)
		: m_quality(a.m_quality), m_uid(a.m_uid), m_instrumentName(a.m_instrumentName)
	{
		m_matrix = a.m_matrix;
	}


	TrackerData& TrackerData::operator=(const TrackerData& a)
	{
		if (this != &a) {
			m_matrix = a.m_matrix;
			m_quality = a.m_quality;
			m_uid = a.m_uid;
			m_instrumentName = a.m_instrumentName;
		}

		return *this;
	}

	void TrackerData::setPositionQuaternion(const std::vector<double>& pos, const std::vector<double>& quat)
	{
		// Following http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/geometric/orthogonal/index.htm
		// If a quaternion is represented by qw + i qx + j qy + k qz , then the equivalent matrix, 
		// to represent the same rotation, is:
		// 1 - 2*qy2 - 2*qz2 	2*qx*qy - 2*qz*qw 	2*qx*qz + 2*qy*qw
		// 2*qx*qy + 2*qz*qw 	1 - 2*qx2 - 2*qz2 	2*qy*qz - 2*qx*qw
		// 2*qx*qz - 2*qy*qw 	2*qy*qz + 2*qx*qw 	1 - 2*qx2 - 2*qy2

		assert(pos.size() == 3 && quat.size() == 4);

		double qw = quat[0];
		double qx = quat[1];
		double qy = quat[2];
		double qz = quat[3];

		// calculate rotation matrix and store it
		m_matrix[0] = 1 - 2 * qy*qy - 2 * qz*qz;
		m_matrix[1] = 2 * qx*qy - 2 * qz*qw;
		m_matrix[2] = 2 * qx*qz + 2 * qy*qw;

		m_matrix[4] = 2 * qx*qy + 2 * qz*qw;
		m_matrix[5] = 1 - 2 * qx*qx - 2 * qz*qz;
		m_matrix[6] = 2 * qy*qz - 2 * qx*qw;

		m_matrix[8] = 2 * qx*qz - 2 * qy*qw;
		m_matrix[9] = 2 * qy*qz + 2 * qx*qw;
		m_matrix[10] = 1 - 2 * qx*qx - 2 * qy*qy;

		// fill in position information
		m_matrix[3] = pos[0];
		m_matrix[7] = pos[1];
		m_matrix[11] = pos[2];
		m_matrix[12] = 0;
		m_matrix[13] = 0;
		m_matrix[14] = 0;
		m_matrix[15] = 1;
	};

	void TrackerData::print()
	{
		std::string outStr = "\n";

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++)
			{
				outStr += "\t" + std::to_string(m_matrix[i * 4 + j]);
			}
			outStr += "\n";
		}

		logging::log_log(outStr, "\t", m_quality);
	}
}