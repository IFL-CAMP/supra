// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "TimeGainCompensationNode.h"
#include "TimeGainCompensation.h"

#include "USImage.h"
#include "Beamformer/USRawData.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	TimeGainCompensationNode::TimeGainCompensationNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_factor(1.0)
		, m_compensator(std::make_shared<TimeGainCompensation>())
	{
		// Create the underlying tbb node for handling the message passing. This usually does not need to be modified.
		if (queueing)
		{
			m_node = unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndProcess(inObj); }));
		}
		else
		{
			m_node = unique_ptr<NodeTypeDiscarding>(
				new NodeTypeDiscarding(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndProcess(inObj); }));
		}

		m_callFrequency.setName("TimeGainCompensationNode");

		// Define the parameters that this node reveals to the user
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeUint8, TypeInt16 }, TypeFloat, "Output type");
		m_valueRangeDictionary.set<double>("depth0", 0.0, 1.0, 0.0, "Depth 0");
		m_valueRangeDictionary.set<double>("level0", -70.0, 70.0, 0.0, "Level 0 [dB]");
		m_valueRangeDictionary.set<double>("depth1", 0.0, 1.0, 0.11, "Depth 1");
		m_valueRangeDictionary.set<double>("level1", -70.0, 70.0, 0.0, "Level 1 [dB]");
		m_valueRangeDictionary.set<double>("depth2", 0.0, 1.0, 0.22, "Depth 2");
		m_valueRangeDictionary.set<double>("level2", -70.0, 70.0, 0.0, "Level 2 [dB]");
		m_valueRangeDictionary.set<double>("depth3", 0.0, 1.0, 0.33, "Depth 3");
		m_valueRangeDictionary.set<double>("level3", -70.0, 70.0, 0.0, "Level 3 [dB]");
		m_valueRangeDictionary.set<double>("depth4", 0.0, 1.0, 0.44, "Depth 4");
		m_valueRangeDictionary.set<double>("level4", -70.0, 70.0, 0.0, "Level 4 [dB]");
		m_valueRangeDictionary.set<double>("depth5", 0.0, 1.0, 0.56, "Depth 5");
		m_valueRangeDictionary.set<double>("level5", -70.0, 70.0, 0.0, "Level 5 [dB]");
		m_valueRangeDictionary.set<double>("depth6", 0.0, 1.0, 0.67, "Depth 6");
		m_valueRangeDictionary.set<double>("level6", -70.0, 70.0, 0.0, "Level 6 [dB]");
		m_valueRangeDictionary.set<double>("depth7", 0.0, 1.0, 0.78, "Depth 7");
		m_valueRangeDictionary.set<double>("level7", -70.0, 70.0, 0.0, "Level 7 [dB]");
		m_valueRangeDictionary.set<double>("depth8", 0.0, 1.0, 0.89, "Depth 8");
		m_valueRangeDictionary.set<double>("level8", -70.0, 70.0, 0.0, "Level 8 [dB]");
		m_valueRangeDictionary.set<double>("depth9", 0.0, 1.0, 1.0, "Depth 9");
		m_valueRangeDictionary.set<double>("level9", -70.0, 70.0, 0.0, "Level 9 [dB]");

		// read the configuration to apply the default values
		configurationChanged();
	}

	void TimeGainCompensationNode::configurationChanged()
	{
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
		readAndSetCurvePoints();
	}

	void TimeGainCompensationNode::configurationEntryChanged(const std::string& configKey)
	{
		// lock the object mutex to make sure no processing happens during parameter changes
		unique_lock<mutex> l(m_mutex);
		if (configKey == "factor")
		{
			m_factor = m_configurationDictionary.get<double>("factor");
		}
		else if (configKey == "outputType")
		{
			m_outputType = m_configurationDictionary.get<DataType>("outputType");
		}
		else if (configKey.compare(0, 5, "depth") == 0 || configKey.compare(0, 5, "level") == 0)
		{
			readAndSetCurvePoints();
		}
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> TimeGainCompensationNode::processTemplateSelection(
	        std::shared_ptr<const Container<InputType> > imageData, vec3s size, size_t workDimension)
	{
		// With the function already templated on the input type, handle the desired output type.
		switch (m_outputType)
		{
		case supra::TypeUint8:
			return m_compensator->process<InputType, uint8_t>(imageData, size, workDimension);
			break;
		case supra::TypeInt16:
			return m_compensator->process<InputType, int16_t>(imageData, size, workDimension);
			break;
		case supra::TypeFloat:
			return m_compensator->process<InputType, float>(imageData, size, workDimension);
			break;
		default:
			logging::log_error("TimeGainCompensationNode: Output image type not supported");
			break;
		}
		return nullptr;
	}

	shared_ptr<RecordObject> TimeGainCompensationNode::checkTypeAndProcess(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<RecordObject> pOut = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
			if (pInImage)
			{
				// lock the object mutex to make sure no parameters are changed during processing
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();

				std::shared_ptr<ContainerBase> pProcessedData;

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types, 
				// only those meaningful for the operation of the node.
				switch (pInImage->getDataType())
				{
				case TypeUint8:
					pProcessedData = processTemplateSelection<uint8_t>(pInImage->getData<uint8_t>(), pInImage->getSize(), 1);
					break;
				case TypeInt16:
					pProcessedData = processTemplateSelection<int16_t>(pInImage->getData<int16_t>(), pInImage->getSize(), 1);
					break;
				case TypeFloat:
					pProcessedData = processTemplateSelection<float>(pInImage->getData<float>(), pInImage->getSize(), 1);
					break;
				default:
					logging::log_error("TimeGainCompensationNode: Input image type not supported");
					break;
				}
				m_callFrequency.measureEnd();

				// Wrap the returned Container in an USImage with the same size etc.
				pOut = make_shared<USImage>(
					pInImage->getSize(),
					pProcessedData,
					pInImage->getImageProperties(),
					pInImage->getReceiveTimestamp(),
					pInImage->getSyncTimestamp());
			}
			else {
				logging::log_error("TimeGainCompensationNode: could not cast object to USImage type, although its type is TypeUSImage.");
			}
		}
		else if (inObj && inObj->getType() == TypeUSRawData)
		{
			shared_ptr<USRawData> pInRawData = dynamic_pointer_cast<USRawData>(inObj);
			if (pInRawData)
			{
				// lock the object mutex to make sure no parameters are changed during processing
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();

				std::shared_ptr<ContainerBase> pProcessedData;

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types,
				// only those meaningful for the operation of the node.
				vec3s size{pInRawData->getNumSamples(), pInRawData->getNumReceivedChannels(), pInRawData->getNumScanlines()};
				switch (pInRawData->getDataType())
				{
					case TypeUint8:
						pProcessedData = processTemplateSelection<uint8_t>(pInRawData->getData<uint8_t>(), size, 0);
						break;
					case TypeInt16:
						pProcessedData = processTemplateSelection<int16_t>(pInRawData->getData<int16_t>(), size, 0);
						break;
					case TypeFloat:
						pProcessedData = processTemplateSelection<float>(pInRawData->getData<float>(), size, 0);
						break;
					default:
						logging::log_error("TimeGainCompensationNode: Input image type not supported");
						break;
				}
				m_callFrequency.measureEnd();

				// Wrap the returned Container in an USRawData with the same size etc.
				pOut = make_shared<USRawData>(
						pInRawData->getNumScanlines(),
						pInRawData->getNumElements(),
						pInRawData->getElementLayout(),
						pInRawData->getNumReceivedChannels(),
						pInRawData->getNumSamples(),
						pInRawData->getSamplingFrequency(),
						pProcessedData,
						pInRawData->getRxBeamformerParameters(),
						pInRawData->getImageProperties(),
						pInRawData->getReceiveTimestamp(),
						pInRawData->getSyncTimestamp());
			}
			else {
				logging::log_error("TimeGainCompensationNode: could not cast object to USRawData type, although its type is TypeUSRawData.");
			}
		}
		return pOut;
	}

	void TimeGainCompensationNode::readAndSetCurvePoints() {
		std::vector<std::pair<double, double> > curvePoints(10);
		curvePoints[0] = std::make_pair(m_configurationDictionary.get<double>("depth0"), m_configurationDictionary.get<double>("level0"));
		curvePoints[1] = std::make_pair(m_configurationDictionary.get<double>("depth1"), m_configurationDictionary.get<double>("level1"));
		curvePoints[2] = std::make_pair(m_configurationDictionary.get<double>("depth2"), m_configurationDictionary.get<double>("level2"));
		curvePoints[3] = std::make_pair(m_configurationDictionary.get<double>("depth3"), m_configurationDictionary.get<double>("level3"));
		curvePoints[4] = std::make_pair(m_configurationDictionary.get<double>("depth4"), m_configurationDictionary.get<double>("level4"));
		curvePoints[5] = std::make_pair(m_configurationDictionary.get<double>("depth5"), m_configurationDictionary.get<double>("level5"));
		curvePoints[6] = std::make_pair(m_configurationDictionary.get<double>("depth6"), m_configurationDictionary.get<double>("level6"));
		curvePoints[7] = std::make_pair(m_configurationDictionary.get<double>("depth7"), m_configurationDictionary.get<double>("level7"));
		curvePoints[8] = std::make_pair(m_configurationDictionary.get<double>("depth8"), m_configurationDictionary.get<double>("level8"));
		curvePoints[9] = std::make_pair(m_configurationDictionary.get<double>("depth9"), m_configurationDictionary.get<double>("level9"));

		m_compensator->setCurve(curvePoints);
	}
}