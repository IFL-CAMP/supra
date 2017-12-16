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

#ifndef __METAIMAGEOUTPUTDEVICE_H__
#define __METAIMAGEOUTPUTDEVICE_H__

#ifdef HAVE_DEVICE_METAIMAGE_OUTPUT

#include "AbstractOutput.h"

namespace supra
{
	//forward declarations
	class MhdSequenceWriter;
	template <typename T>
	class USImage;
	template <typename T>
	class USRawData;

	class MetaImageOutputDevice : public AbstractOutput
	{
	public:
		MetaImageOutputDevice(tbb::flow::graph& graph, const std::string & nodeID);
		~MetaImageOutputDevice();

		//Functions to be overwritten
	public:
		virtual void initializeOutput();
		virtual bool ready();
		//Needs to be thread safe
		virtual void startSequence();
		//Needs to be thread safe
		virtual void stopSequence();
	protected:
		virtual void startOutput();
		//Needs to be thread safe
		virtual void stopOutput();
		//Needs to be thread safe
		virtual void configurationDone();

		virtual void writeData(std::shared_ptr<RecordObject> data);

	private:
		void initializeSequence();
		void addData(std::shared_ptr<const RecordObject> data);
		void addSyncRecord(std::shared_ptr<const RecordObject> _syncMessage);
		size_t addImage(std::shared_ptr<const RecordObject> imageData);
		size_t addUSRawData(std::shared_ptr<const RecordObject> _rawData);
		void addTracking(std::shared_ptr<const RecordObject> trackingData, size_t frameNum);

		template <typename ElementType>
		size_t addImageTemplated(std::shared_ptr<const USImage<ElementType> > imageData);
		template <typename ElementType>
		size_t addUSRawDataTemplated(std::shared_ptr < const USRawData<ElementType> > rawData);

		MhdSequenceWriter* m_pWriter;

		bool m_isReady;
		bool m_createSequences;
		bool m_active;
		std::atomic_bool m_isRecording;
		size_t m_sequencesWritten;

		std::mutex m_writerMutex;

		std::string m_filename;
	};
}

#endif //!HAVE_DEVICE_METAIMAGE_OUTPUT

#endif //!__METAIMAGEOUTPUTDEVICE_H__
