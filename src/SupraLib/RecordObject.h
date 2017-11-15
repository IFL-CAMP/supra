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

#ifndef __RECORDOBJECT_H__
#define __RECORDOBJECT_H__

namespace supra
{
	/// Enum for the different types of data distributed within the dataflow graph
	enum RecordObjectType {
		TypeSyncRecordObject,
		TypeTrackerDataSet,
		TypeUSImage,
		TypeUSRawData,
		TypeRecordUnknown
	};

	//const char* RecordObjectTypeToString(RecordObjectType t);

	/// Base class for all RecordObjects
	class RecordObject
	{

	public:
		/// Constructor that takes the two timestamps:
		///		receiveTimestamp is the timestamp (in seconds) at which this dataset initially was recieved
		///		syncTimestamp is the timestamp (in seconds) after optinal temporal synchronization
		RecordObject(double receiveTimestamp, double syncTimestamp) : m_receiveTimestamp(receiveTimestamp), m_syncTimestamp(syncTimestamp) {};
		/// Base constructor
		RecordObject() : m_receiveTimestamp(0.0), m_syncTimestamp(0.0) {};
		virtual ~RecordObject() {};

		/// Returns the receive timestamp (in seconds), that is the time at which this dataset initially was recieved
		inline double getReceiveTimestamp() const { return m_receiveTimestamp; };
		/// Returns the snyc timestamp (in seconds), that is after optinal temporal synchronization
		inline double getSyncTimestamp() const { return m_syncTimestamp; };

		/// Returns the type of the dataset. Overwritten in subclasses.
		virtual RecordObjectType getType() const { return TypeRecordUnknown; }

	protected:
		/// The receive timestamp(in seconds), that is the time at which this dataset initially was recieved
		double m_syncTimestamp;
		/// The snyc timestamp (in seconds), that is after optinal temporal synchronization
		double m_receiveTimestamp;
	};
}

#endif // !__RECORDOBJECT_H__
