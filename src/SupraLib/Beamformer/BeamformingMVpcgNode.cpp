// ================================================================================================
// 
// Copyright (C) 2017, Rüdiger Göbl - all rights reserved
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
//          Rüdiger Göbl
//          Email r.goebl@tum.de
//          Chair for Computer Aided Medical Procedures
//          Technische Universität München
//          Boltzmannstr. 3, 85748 Garching b. München, Germany
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

#include "BeamformingMVpcgNode.h"
#include "RxBeamformerMVpcg.h"

#include "USImage.h"
#include "USRawData.h"

#include <utilities/Logging.h>
#include <utilities/cudaUtility.h>
using namespace std;
using namespace supra::RxBeamformerMVpcg;

namespace supra
{
	BeamformingMVpcgNode::BeamformingMVpcgNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_lastSeenImageProperties(nullptr)
		, m_outputType(TypeUnknown)
	{
		if (queueing)
		{
			m_node = unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); }));
		}
		else
		{
			m_node = unique_ptr<NodeTypeDiscarding>(
				new NodeTypeDiscarding(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); }));
		}

		m_callFrequency.setName("BeamformingMV");
		m_valueRangeDictionary.set<uint32_t>("subArraySize", 0, 64, 56, "Sub-array size");
		m_valueRangeDictionary.set<uint32_t>("temporalSmoothing", 0, 500, 10, "temporal smoothing");
		m_valueRangeDictionary.set<uint32_t>("maxIterationsOverride", 0, 10000, 0, "Max iterations override (if != 0)");
		m_valueRangeDictionary.set<double>("convergenceThresholdExponent", -100, 0, -100, "Convergence threshold exponent");
		m_valueRangeDictionary.set<double>("convergenceThreshold", 0.0, 1, 1e-15, "solver convergence Threshold");
		m_valueRangeDictionary.set<double>("outputClamp", 0.0, 1e30, 100, "output clamp");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeInt16 }, TypeFloat, "Output type");
		m_valueRangeDictionary.set<double>("subArrayScalingPower", 0.5, 3.0, 1.5, "Subarray count scaling power");
		
		configurationChanged();
	}

	BeamformingMVpcgNode::~BeamformingMVpcgNode()
	{
	}

	void BeamformingMVpcgNode::configurationChanged()
	{
		m_subArraySize = m_configurationDictionary.get<uint32_t>("subArraySize");
		m_temporalSmoothing = m_configurationDictionary.get<uint32_t>("temporalSmoothing");
		m_maxIterations = m_configurationDictionary.get<uint32_t>("maxIterations");
		m_convergenceThreshold = m_configurationDictionary.get<double>("convergenceThreshold");
		m_outputClamp = m_configurationDictionary.get<double>("outputClamp");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
		m_maxIterationsOverride = m_configurationDictionary.get<uint32_t>("maxIterationsOverride");
		m_convergenceThreshold = std::pow(10, m_configurationDictionary.get<double>("convergenceThresholdExponent"));
		m_subArrayScalingPower = m_configurationDictionary.get<double>("subArrayScalingPower");
	}

	void BeamformingMVpcgNode::configurationEntryChanged(const std::string& configKey)
	{
		unique_lock<mutex> l(m_mutex);
		if (configKey == "subArraySize")
		{
			m_subArraySize = m_configurationDictionary.get<uint32_t>("subArraySize");
		}
		else if (configKey == "temporalSmoothing")
		{
			m_temporalSmoothing = m_configurationDictionary.get<uint32_t>("temporalSmoothing");
		}
		else if (configKey == "maxIterations")
		{
			m_maxIterations = m_configurationDictionary.get<uint32_t>("maxIterations");
		}
		else if (configKey == "convergenceThreshold")
		{
			m_convergenceThreshold = m_configurationDictionary.get<double>("convergenceThreshold");
		}
		else if (configKey == "outputClamp")
		{
			m_outputClamp = m_configurationDictionary.get<double>("outputClamp");
		}
		else if (configKey == "outputType")
		{
			m_outputType = m_configurationDictionary.get<DataType>("outputType");
		}
		else if (configKey == "maxIterationsOverride")
		{
			m_maxIterationsOverride = m_configurationDictionary.get<uint32_t>("maxIterationsOverride");
		}
		else if (configKey == "convergenceThresholdExponent")
		{
			m_convergenceThreshold = std::pow(10, m_configurationDictionary.get<double>("convergenceThresholdExponent"));
		}
		else if (configKey == "subArrayScalingPower")
		{
			m_subArrayScalingPower = m_configurationDictionary.get<double>("subArrayScalingPower");
		}
		if (m_lastSeenImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}

	template <typename RawDataType>
	std::shared_ptr<USImage> BeamformingMVpcgNode::beamformTemplated(shared_ptr<const USRawData> rawData)
	{
		shared_ptr<USImage> pImageRF = nullptr;
		switch (m_outputType)
		{
		case supra::TypeInt16:
			pImageRF = performRxBeamforming<RawDataType, int16_t>(
				rawData, m_subArraySize, m_temporalSmoothing, m_maxIterationsOverride, m_convergenceThreshold, m_subArrayScalingPower, m_outputClamp);
			break;
		case supra::TypeFloat:
			pImageRF = performRxBeamforming<RawDataType, float>(
				rawData, m_subArraySize, m_temporalSmoothing, m_maxIterationsOverride, m_convergenceThreshold, m_subArrayScalingPower, m_outputClamp);
			break;
		default:
			logging::log_error("BeamformingMVNode: Output image type not supported:");
			break;
		}
		return pImageRF;
	}

	shared_ptr<RecordObject> BeamformingMVpcgNode::checkTypeAndBeamform(shared_ptr<RecordObject> inObj)
	{
		unique_lock<mutex> l(m_mutex);

		shared_ptr<USImage> pImageRF = nullptr;
		if (inObj->getType() == TypeUSRawData)
		{
			shared_ptr<const USRawData> pRawData = dynamic_pointer_cast<const USRawData>(inObj);
			if (pRawData)
			{
				if (pRawData->getImageProperties()->getImageState() == USImageProperties::RawDelayed)
				{
					m_callFrequency.measure();
					switch (pRawData->getDataType())
					{
					case TypeInt16:
						pImageRF = beamformTemplated<int16_t>(pRawData);
						break;
					case TypeFloat:
						pImageRF = beamformTemplated<float>(pRawData);
						break;
					default:
						logging::log_error("BeamformingMVNode: Input rawdata type is not supported.");
						break;
					}
					m_callFrequency.measureEnd();

					if (m_lastSeenImageProperties != pImageRF->getImageProperties())
					{
						updateImageProperties(pImageRF->getImageProperties());
					}
					pImageRF->setImageProperties(m_editedImageProperties);
				}
				else {
					logging::log_error("BeamformingMVNode: Cannot beamform undelayed RawData. Apply RawDelayNode first");
				}
			}
			else {
				logging::log_error("BeamformingMVNode: could not cast object to USRawData type, is it in supported ElementType?");
			}
		}
		return pImageRF;
	}

	void BeamformingMVpcgNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		m_editedImageProperties = make_shared<USImageProperties>(*imageProperties);
		m_editedImageProperties->setImageState(USImageProperties::RF);
		m_editedImageProperties->setSpecificParameter("BeamformingMVNode.subArraySize", m_subArraySize);
		m_editedImageProperties->setSpecificParameter("BeamformingMVNode.temporalSmoothing", m_temporalSmoothing);
	}
}
