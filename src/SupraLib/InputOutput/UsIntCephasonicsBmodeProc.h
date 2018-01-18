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


#ifndef __USINTCEPHASONICSBMODEPROC_H__
#define __USINTCEPHASONICSBMODEPROC_H__

#include <DataProcessor.h>
#include <atomic>

namespace supra
{
	class UsIntCephasonicsBmode;

	class UsIntCephasonicsBmodeProc : public DataProcessor
	{
	public:
		UsIntCephasonicsBmodeProc(
			PlatformHandle& handle,
			UsIntCephasonicsBmode* inputNode);

		virtual int process(ScanData* data);
		int layoutChanged(ImageLayout& layout);
		void setActiveStage(int activeStage);
	private:
		ImageLayout m_imageLayout;
		UsIntCephasonicsBmode* m_pInputNode;
		std::atomic<int> m_activeStage;
	};
}

#endif //!__USINTCEPHASONICSBMODEPROC_H__
