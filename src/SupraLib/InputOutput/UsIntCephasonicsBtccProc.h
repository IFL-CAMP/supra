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


#ifndef __USINTCEPHASONICSBTCCPROC_H__
#define __USINTCEPHASONICSBTCCPROC_H__

#include <DataProcessor.h>

namespace supra
{
	class UsIntCephasonicsBtcc;

	class UsIntCephasonicsBtccProc : public DataProcessor
	{
	public:
		UsIntCephasonicsBtccProc(
			PlatformHandle& handle,
			UsIntCephasonicsBtcc* inputNode);

		virtual int process(ScanData* data);
		int layoutChanged(ImageLayout& layout);

	private:
		ImageLayout m_imageLayout;
		UsIntCephasonicsBtcc* m_pInputNode;
	};
}

#endif //!__USINTCEPHASONICSBTCCPROC_H__
