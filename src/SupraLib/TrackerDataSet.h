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

#ifndef __TRACKERDATASET_H__
#define __TRACKERDATASET_H__

#include "TrackerData.h"
#include "RecordObject.h"

#include <vector>

namespace supra
{
	/// A compute graph object that represents a collection of 
	/// tracker measurements (from the same point in time)
	class TrackerDataSet : public RecordObject
	{
	public:
		/// Base constructor
		TrackerDataSet();
		/// Constructs a TrackerDataset with a vector of \see TrackerData objects, each representing
		/// a tracker measurement (all of measurements are from the same point in time)
		TrackerDataSet(std::vector<TrackerData>& sensorData, double receiveTimestamp, double syncTimestamp);
		/// Copy constructor, copies tracker data and metadata
		TrackerDataSet(const TrackerDataSet& set);

		/// Assignment operator, copies tracker data and metadata
		TrackerDataSet& operator=(const TrackerDataSet& a);

		/// Returns the vector of TrackerData containing the pose measurements
		inline const std::vector<TrackerData>& getSensorData() const { return m_sensorData; };

		//void getHeaderDescriptor(std::vector<HeaderDescriptor>& descriptors);

		//int getSize (const unsigned encoding) const;

		//void writeToStream(std::ostream* outputStream, uint32_t encoding);
		//void getFromStream(std::istream* inputStream, uint32_t encoding);

		virtual RecordObjectType getType() const { return TypeTrackerDataSet; }

	protected:
		/// The vector of TrackerData containing the pose measurements
		std::vector<TrackerData> m_sensorData;
	};
}

#endif //!__TRACKERDATASET_H__
