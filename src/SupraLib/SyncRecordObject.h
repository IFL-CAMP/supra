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

#ifndef __SYNCRECORDOBJECT_H__
#define __SYNCRECORDOBJECT_H__

#include "RecordObject.h"

#include <memory>
#include <vector>

namespace supra
{
	/// A dataflow graph object that contains several \see RecordObject after synchronization.
	/// This is for example used to synchonize tracker measurements to an ultrasound image stream.
	/// For that, one RecordObject is considered the main record and the synchronized records
	/// are selected (by \see StreamSynchonizer ) to match the main record's timestamp as close
	/// as possible.
	class SyncRecordObject : public RecordObject
	{

	public:
		/// Creates a synchronized record from a main record and a vector of synched records
		SyncRecordObject(
			std::shared_ptr<const RecordObject> mainRecord,
			std::vector<std::shared_ptr<const RecordObject> > syncedRecords,
			double receiveTimestamp, double syncTimestamp)
			: RecordObject(receiveTimestamp, syncTimestamp)
			, m_pMainRecord(mainRecord)
			, m_pSyncedRecords(syncedRecords) {};
		virtual ~SyncRecordObject() {};

		virtual RecordObjectType getType() const { return TypeSyncRecordObject; }

		/// Returns the main record
		std::shared_ptr<const RecordObject> getMainRecord() const { return m_pMainRecord; }
		/// Returns the synchonized records
		std::vector<std::shared_ptr<const RecordObject> > getSyncedRecords() const { return m_pSyncedRecords; };

	private:
		std::shared_ptr<const RecordObject> m_pMainRecord;
		std::vector<std::shared_ptr<const RecordObject> > m_pSyncedRecords;
	};
}

#endif // !__SYNCRECORDOBJECT_H__