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

#include <DataProcessor.h>
#include "UsIntCephasonicsCc.h"

#include <ScanData.h>

#include "UsIntCephasonicsCcProc.h"

#include <iomanip>

namespace supra
{
	using namespace cs;

	UsIntCephasonicsCcProc::UsIntCephasonicsCcProc(
		PlatformHandle& handle,
		UsIntCephasonicsCc* inputNode)
		: DataProcessor(handle, false)
		, m_pInputNode(inputNode)
	{
		skipStorageStage(RAW, true);
		skipStorageStage(STAGE0, true);
		skipStorageStage(STAGE1, true);
		skipStorageStage(STAGE2, true);
		skipStorageStage(STAGE3, true);
		skipStorageStage(STAGE4, true);
		skipStorageStage(STAGE5, true);

	}

	int UsIntCephasonicsCcProc::process(ScanData * data)
	{
		if (data->mode == cs::CHCAP)
		{
			/*printf("------------------------------------\nScanData:   seqNo: %d platformIdx: %d numBeams: %d samplesPerBeam: %d size: %d\n",
			data->seqNo,
			data->platformIdx,
			data->numBeams,
			data->samplesPerBeam,
			data->size);*/
			size_t numSamples = data->samplesPerBeam;
			size_t numChannels = data->bytesPerSample * 8 / 12; // the 64 channels that are acquired at the same time
			numChannels = 64;
			//each with 12 bit in not unscrambled mode
			size_t numBeams = data->numBeams;

			// unique ID for SubFrameDef within Sequence
			size_t frameID = data->sframeID; 

			// frame count to tbe used for drop frames
			size_t frameNo = data->seqNo;

			// forward data to processor
			m_pInputNode->putData(data->platformIdx, frameID, frameNo, numChannels, numSamples, numBeams, data->data);

			data->markRelease();

			//printf("seq: %d sframe: %d pltform: %d\n", frameNo, frameID, static_cast<int>(data->platformIdx));
		}
		return CS_SUCCESS;
	}

	int UsIntCephasonicsCcProc::layoutChanged(ImageLayout & layout)
	{
		m_imageLayout = ImageLayout(layout);
		m_pInputNode->layoutChanged(layout);
		return CS_SUCCESS;
	}
}
