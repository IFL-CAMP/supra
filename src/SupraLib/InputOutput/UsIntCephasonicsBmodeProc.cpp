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
#include "UsIntCephasonicsBmode.h"

#include "UsIntCephasonicsBmodeProc.h"
#include "utilities/Logging.h"

namespace supra
{
	using namespace ::cs;

	UsIntCephasonicsBmodeProc::UsIntCephasonicsBmodeProc(
		PlatformHandle& handle,
		UsIntCephasonicsBmode* inputNode)
		: DataProcessor(handle, false) // Second argument to constructor is defaultDataPath
		, m_pInputNode(inputNode)
		, m_activeStage(4)
	{
	}

	int UsIntCephasonicsBmodeProc::process(ScanData * data)
	{
		//Stage 1
		if (m_activeStage == 0)
		{
			/*logging::log_log("Stage ", data->postProcInfo->stage, " {",
				" seqNo ", data->seqNo,
				" mode ", data->mode,
				" numBeams ", data->numBeams,
				" numParallelBeams ", data->numParallelBeams,
				" samplesPerBeam ", data->samplesPerBeam,
				" bytesPerSample ", data->bytesPerSample,
				" sframeID ", data->sframeID,
				" cibTrigCount ", data->cibTrigCount,
				" hdrsz ", data->hdrsz,
				" size ", data->size,
				" platformIdx ", data->platformIdx,
				" postProcInfo ", data->postProcInfo,
				" data ", data->data,
				" fb ", data->fb,
				" _bufferpoolF ", data->_bufferpoolF, "}");*/
			if (data->fb)
			{	
				FrameBuffer* pFrameBuffer = data->fb;
				m_pInputNode->putData(pFrameBuffer);
			}
			else
			{
				m_pInputNode->putData(data);
			}
		}
		processStage1(data);

		//Stage 2
		if (m_activeStage == 1)
		{
			/*logging::log_log("Stage ", data->postProcInfo->stage, " {",
				" seqNo ", data->seqNo,
				" mode ", data->mode,
				" numBeams ", data->numBeams,
				" numParallelBeams ", data->numParallelBeams,
				" samplesPerBeam ", data->samplesPerBeam,
				" bytesPerSample ", data->bytesPerSample,
				" sframeID ", data->sframeID,
				" cibTrigCount ", data->cibTrigCount,
				" hdrsz ", data->hdrsz,
				" size ", data->size,
				" platformIdx ", data->platformIdx,
				" postProcInfo ", data->postProcInfo,
				" data ", data->data,
				" fb ", data->fb,
				" _bufferpoolF ", data->_bufferpoolF, "}");*/
			if (data->fb)
			{	
				FrameBuffer* pFrameBuffer = data->fb;
				m_pInputNode->putData(pFrameBuffer);
			}
			else
			{
				m_pInputNode->putData(data);
			}
		}
		processStage2(data);

		//Stage 3
		if (m_activeStage == 2)
		{
			/*logging::log_log("Stage ", data->postProcInfo->stage, " {",
				" seqNo ", data->seqNo,
				" mode ", data->mode,
				" numBeams ", data->numBeams,
				" numParallelBeams ", data->numParallelBeams,
				" samplesPerBeam ", data->samplesPerBeam,
				" bytesPerSample ", data->bytesPerSample,
				" sframeID ", data->sframeID,
				" cibTrigCount ", data->cibTrigCount,
				" hdrsz ", data->hdrsz,
				" size ", data->size,
				" platformIdx ", data->platformIdx,
				" postProcInfo ", data->postProcInfo,
				" data ", data->data,
				" fb ", data->fb,
				" _bufferpoolF ", data->_bufferpoolF, "}");*/
			if (data->fb)
			{	
				FrameBuffer* pFrameBuffer = data->fb;
				m_pInputNode->putData(pFrameBuffer);
			}
			else
			{
				m_pInputNode->putData(data);
			}
		}
		processStage3(data);

		//Stage 4
		if (m_activeStage == 3)
		{
			/*logging::log_log("Stage ", data->postProcInfo->stage, " {",
				" seqNo ", data->seqNo,
				" mode ", data->mode,
				" numBeams ", data->numBeams,
				" numParallelBeams ", data->numParallelBeams,
				" samplesPerBeam ", data->samplesPerBeam,
				" bytesPerSample ", data->bytesPerSample,
				" sframeID ", data->sframeID,
				" cibTrigCount ", data->cibTrigCount,
				" hdrsz ", data->hdrsz,
				" size ", data->size,
				" platformIdx ", data->platformIdx,
				" postProcInfo ", data->postProcInfo,
				" data ", data->data,
				" fb ", data->fb,
				" _bufferpoolF ", data->_bufferpoolF, "}");*/
			if (data->fb)
			{	
				FrameBuffer* pFrameBuffer = data->fb;
				m_pInputNode->putData(pFrameBuffer);
			}
			else
			{
				m_pInputNode->putData(data);
			}
		}
		processStage4(data);

		//Stage 5
		if (m_activeStage == 4)
		{
			/*logging::log_log("Stage ", data->postProcInfo->stage, " {",
				" seqNo ", data->seqNo,
				" mode ", data->mode,
				" numBeams ", data->numBeams,
				" numParallelBeams ", data->numParallelBeams,
				" samplesPerBeam ", data->samplesPerBeam,
				" bytesPerSample ", data->bytesPerSample,
				" sframeID ", data->sframeID,
				" cibTrigCount ", data->cibTrigCount,
				" hdrsz ", data->hdrsz,
				" size ", data->size,
				" platformIdx ", data->platformIdx,
				" postProcInfo ", data->postProcInfo,
				" data ", data->data,
				" fb ", data->fb,
				" _bufferpoolF ", data->_bufferpoolF, "}");*/
			if (data->fb)
			{	
				FrameBuffer* pFrameBuffer = data->fb;
				m_pInputNode->putData(pFrameBuffer);
			}
			else
			{
				m_pInputNode->putData(data);
			}
		}
		return CS_SUCCESS;
	}

	int UsIntCephasonicsBmodeProc::layoutChanged(ImageLayout & layout)
	{
		m_imageLayout = ImageLayout(layout);
		m_pInputNode->layoutChanged(layout);
		return CS_SUCCESS;
	}

	void UsIntCephasonicsBmodeProc::setActiveStage(int activeStage)
	{
		m_activeStage = activeStage;
	}
}
