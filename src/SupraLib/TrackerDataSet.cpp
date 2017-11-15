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

#include "TrackerDataSet.h"

namespace supra
{
	TrackerDataSet::TrackerDataSet() : RecordObject() { }

	TrackerDataSet::TrackerDataSet(std::vector<TrackerData>& sensorData, double receiveTimestamp, double syncTimestamp)
		: m_sensorData(sensorData), RecordObject(receiveTimestamp, syncTimestamp) { }


	TrackerDataSet::TrackerDataSet(const TrackerDataSet& a)
		: m_sensorData(a.m_sensorData), RecordObject(a.m_receiveTimestamp, a.m_syncTimestamp) { }


	TrackerDataSet& TrackerDataSet::operator=(const TrackerDataSet& a)
	{
		if (this != &a) {
			m_sensorData = a.m_sensorData;
			m_receiveTimestamp = a.m_receiveTimestamp;
			m_syncTimestamp = a.m_syncTimestamp;
		}

		return *this;
	}


	//void TrackerDataSet::getHeaderDescriptor(std::vector<HeaderDescriptor>& descriptors)
	//{
	//	descriptors.clear();
	//	descriptors.push_back(HeaderDescriptor(dtDOUBLE,1,"SyncTimestamp"));
	//	descriptors.push_back(HeaderDescriptor(dtDOUBLE,1,"FtlTimestamp"));
	//	descriptors.push_back(HeaderDescriptor(dtUINT32,1,"NumberOfSensors"));
	//	for (std::vector<TrackerData>::iterator it = m_sensorData.begin(); it != m_sensorData.end(); ++it)
	//	{
	//		descriptors.push_back(HeaderDescriptor(dtINT32,1,"Uid"));
	//		descriptors.push_back(HeaderDescriptor(dtINT32,1,"Quality"));
	//		descriptors.push_back(HeaderDescriptor(dtDOUBLE,1,"SensorTimestamp"));
	//		descriptors.push_back(HeaderDescriptor(dtDOUBLE,3,"Position"));
	//		descriptors.push_back(HeaderDescriptor(dtDOUBLE,4,"Quaternion"));		
	//	}
	//}

	//int TrackerDataSet::getSize (const unsigned encoding) const
	//{
	//	return sizeof (double)
	//		+ sizeof (double)
	//		+ sizeof (uint32_t)
	//		+ m_sensorData.size ()
	//			* (sizeof (int32_t)
	//				+ sizeof (int32_t)
	//				+ sizeof (double)
	//				+ 3 * sizeof (double)
	//				+ 4 * sizeof (double));
	//}

	//void TrackerDataSet::writeToStream(std::ostream* outputStream, unsigned int encoding)
	//{
	//	uint32_t sensorDataSize = static_cast<uint32_t>(m_sensorData.size());
	//	write<double>(outputStream, &m_syncTimestamp, 1, encoding);
	//	write<double>(outputStream, &m_ftlTimestamp, 1, encoding);
	//	write<uint32_t>(outputStream, &sensorDataSize, 1, encoding);
	//
	//	for (std::vector<TrackerData>::iterator it = m_sensorData.begin(); it != m_sensorData.end(); ++it)
	//	{
	//		write<int32_t>(outputStream, &(it->m_uid), 1, encoding);
	//		write<int32_t>(outputStream, &(it->m_quality), 1, encoding);
	//		write<double>(outputStream, &(it->m_timestamp), 1, encoding);
	//		write<double>(outputStream, it->m_position, 3, encoding);
	//		write<double>(outputStream, it->m_rotation, 4, encoding);
	//	}
	//    if (encTEXT == encoding) {
	//        *outputStream << std::endl;
	//    }
	//}

	//void TrackerDataSet::getFromStream(std::istream* inputStream, unsigned int encoding)
	//{
	//	uint32_t sensorDataSize;
	//	read<double>(inputStream, &m_syncTimestamp, 1, encoding);
	//	read<double>(inputStream, &m_ftlTimestamp, 1, encoding);
	//	read<uint32_t>(inputStream, &sensorDataSize, 1, encoding);
	//
	//	for (uint32_t k=0; k<sensorDataSize; ++k)
	//	{
	//		int uid, quality;
	//		double ts;
	//		double pos[3];
	//		double rot[4];
	//		read<int32_t>(inputStream, &uid, 1, encoding);
	//		read<int32_t>(inputStream, &quality, 1, encoding);
	//		read<double>(inputStream, &ts, 1, encoding);
	//		read<double>(inputStream, pos, 3, encoding);
	//		read<double>(inputStream, rot, 4, encoding);
	//
	//		m_sensorData.push_back(TrackerData(pos, rot, quality, uid, ts));
	//	}
	//}
}