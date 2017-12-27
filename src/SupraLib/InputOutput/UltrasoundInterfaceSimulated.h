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


#ifndef __ULTRASOUNDHANDLERSIMULATED_H__
#define __ULTRASOUNDHANDLERSIMULATED_H__

#ifdef HAVE_DEVICE_ULTRASOUND_SIM

#include <atomic>
#include <memory>
#include <mutex>

#include <AbstractInput.h>
#include <USImage.h>

namespace supra
{
	class UltrasoundInterfaceSimulated : public AbstractInput<RecordObject>
	{
	public:
		UltrasoundInterfaceSimulated(tbb::flow::graph& graph, const std::string & nodeID);

		//Functions to be overwritten
	public:
		virtual void initializeDevice();
		virtual bool ready();

		virtual std::vector<size_t> getImageOutputPorts() { return{ 0 }; };
		virtual std::vector<size_t> getTrackingOutputPorts() { return{}; };

		virtual void freeze();
		virtual void unfreeze();

	protected:
		virtual void startAcquisition();
		//Needs to be thread safe
		virtual void configurationEntryChanged(const std::string& configKey);
		//Needs to be thread safe
		virtual void configurationChanged();

		virtual bool timerCallback();

	private:
		void readConfiguration();

		int m_frequency;
		size_t m_bmodeNumVectors;
		size_t m_bmodeNumSamples;
		double m_gain;
		double m_depth;
		double m_width;
		std::shared_ptr<USImageProperties> m_pImageProperties;

		std::atomic<bool> m_frozen;

		std::mutex m_objectMutex;
	};
}

#endif //!HAVE_DEVICE_ULTRASOUND_SIM

#endif //!__ULTRASOUNDHANDLERSIMULATED_H__