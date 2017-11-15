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
#include "UsIntCephasonicsBtcc.h"

#include <ScanData.h>

#include "UsIntCephasonicsBtccProc.h"

using namespace cs;

namespace supra
{
	UsIntCephasonicsBtccProc::UsIntCephasonicsBtccProc(
		PlatformHandle& handle,
		UsIntCephasonicsBtcc* inputNode)
		: DataProcessor(handle)
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

	int UsIntCephasonicsBtccProc::process(ScanData * data)
	{
		if (data->mode == cs::CHCAP)
		{
			printf("------------------------------------\nScanData:\nseqNo: %d\nmode: %d\nnumBeams: %d\nnumParallelBeams: %d\nsamplesPerBeam: %d\nbytesPerSample: %d\nsframeID: %d\ncibTrigCount: %d\nhdrsz: %d\nsize: %d\nplatformIdx: %d\npostProcInfo: 0x%lx\ndata: 0x%lx\nfb: 0x%lx\n",
				data->seqNo,
				data->mode,
				data->numBeams,
				data->numParallelBeams,
				data->samplesPerBeam,
				data->bytesPerSample,
				data->sframeID,
				data->cibTrigCount,
				data->hdrsz,
				data->size,
				data->platformIdx,
				reinterpret_cast<size_t>(data->postProcInfo),
				reinterpret_cast<size_t>(data->data),
				reinterpret_cast<size_t>(data->fb));
			size_t numSamples = data->samplesPerBeam; // currently fixed to 2048
			size_t numChannels = data->bytesPerSample / sizeof(int16); // the 64 channels that are acquired at the same time
			size_t numBeams = data->numBeams;

			printf("--------------------------\nRaw: %lu\nStage 0: %lu\nStage 1: %lu\nStage 2: %lu\nStage 3: %lu\nStage 4: %lu\nStage 5: %lu\n",
				this->getStoredData(RAW).size(),
				this->getStoredData(STAGE0).size(),
				this->getStoredData(STAGE1).size(),
				this->getStoredData(STAGE2).size(),
				this->getStoredData(STAGE3).size(),
				this->getStoredData(STAGE4).size(),
				this->getStoredData(STAGE5).size());



			m_pInputNode->putData(data->platformIdx, numChannels, numSamples, numBeams, reinterpret_cast<int16*>(data->data));

			data->markRelease();

			//printf("seq: %d  pltform: %d\n", data->seqNo, data->platformIdx);
		}
		return CS_SUCCESS;
	}

	int UsIntCephasonicsBtccProc::layoutChanged(ImageLayout & layout)
	{
		m_imageLayout = ImageLayout(layout);
		m_pInputNode->layoutChanged(layout);
		return CS_SUCCESS;
	}
}