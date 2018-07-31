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


#ifndef __USINTCEPHASONICSCCPROC_H__
#define __USINTCEPHASONICSCCPROC_H__

#include <DataProcessor.h>

namespace supra
{
	class UsIntCephasonicsCc;

	class UsIntCephasonicsCcProc : public DataProcessor
	{
	public:
		UsIntCephasonicsCcProc(
			PlatformHandle& handle,
			UsIntCephasonicsCc* inputNode);

		virtual int process(ScanData* data);
		int layoutChanged(ImageLayout& layout);

	private:
		ImageLayout m_imageLayout;
		UsIntCephasonicsCc* m_pInputNode;
	};
}

#endif //!__USINTCEPHASONICSCCPROC_H__
