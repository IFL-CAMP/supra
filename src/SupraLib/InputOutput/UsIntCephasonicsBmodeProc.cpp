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

namespace supra
{
	using namespace ::cs;

	UsIntCephasonicsBmodeProc::UsIntCephasonicsBmodeProc(
		PlatformHandle& handle,
		UsIntCephasonicsBmode* inputNode)
		: DataProcessor(handle)
		, m_pInputNode(inputNode)
	{

	}

	int UsIntCephasonicsBmodeProc::process(ScanData * data)
	{
		FrameBuffer* pFrameBuffer = data->fb;
		m_pInputNode->putData(pFrameBuffer);
		return CS_SUCCESS;
	}

	int UsIntCephasonicsBmodeProc::layoutChanged(ImageLayout & layout)
	{
		m_imageLayout = ImageLayout(layout);
		m_pInputNode->layoutChanged(layout);
		return CS_SUCCESS;
	}
}