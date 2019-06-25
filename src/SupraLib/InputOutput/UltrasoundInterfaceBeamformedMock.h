// ================================================================================================
// 
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License, version 2.1, as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this program.  If not, see
// <http://www.gnu.org/licenses/>.
//
// ================================================================================================


#ifndef __ULTRASOUNDINTERFACEBEAMFORMEDMOCK_H__
#define __ULTRASOUNDINTERFACEBEAMFORMEDMOCK_H__

#include <atomic>
#include <memory>
#include <mutex>

#include <AbstractInput.h>
#include <USImage.h>
#include <Container.h>

namespace supra
{
	//forward declaration
	class USRawData;

	class UltrasoundInterfaceBeamformedMock : public AbstractInput
	{
	public:
		UltrasoundInterfaceBeamformedMock(tbb::flow::graph& graph, const std::string & nodeID);

		//Functions to be overwritten
	public:
		virtual void initializeDevice();
		virtual bool ready();

		virtual std::vector<size_t> getImageOutputPorts() { return{}; };
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
		static constexpr size_t m_maxSequenceLength = 20;
		static constexpr size_t m_maxSequenceSizeMb = 512;

		void readConfiguration();
		void readNextFrame();

		std::string m_mockMetadataFilename;
		std::vector<std::string> m_mockDataFilenames;
		bool m_singleImage;
		bool m_streamSequenceOnce;
		double m_frequency;
		std::shared_ptr<USImage> m_protoUSImage;
		std::shared_ptr<Container<int16_t> > m_pMockData;

		std::vector<std::shared_ptr<std::ifstream>> m_mockDataStreams;
		std::vector<std::vector<char> > m_mockDataStramReadBuffers;

		std::vector<size_t> m_sequenceLengths;
		size_t m_sequenceIndex;
		size_t m_frameIndex;
		size_t m_numel;
		std::atomic_bool m_frozen;
		bool m_lastFrame;
		bool m_ready;

		std::mutex m_objectMutex;
	};
}

#endif //!__ULTRASOUNDINTERFACEBEAMFORMEDMOCK_H__
