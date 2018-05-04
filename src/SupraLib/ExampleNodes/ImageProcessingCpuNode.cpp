// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2018, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "ImageProcessingCpuNode.h"

#include "USImage.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	ImageProcessingCpuNode::ImageProcessingCpuNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
		, m_factor(1.0)
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

		m_callFrequency.setName("ImageProcessingCpuNode");

		// Define the parameters that this node reveals to the user
		m_valueRangeDictionary.set<double>("factor", 0.0, 2.0, 1.0, "Factor");
		m_valueRangeDictionary.set<DataType>("outputType", { TypeFloat, TypeUint8, TypeInt16 }, TypeFloat, "Output type");
		
		// read the configuration to apply the default values
		configurationChanged();
	}

	void ImageProcessingCpuNode::configurationChanged()
	{
		m_factor = m_configurationDictionary.get<double>("factor");
		m_outputType = m_configurationDictionary.get<DataType>("outputType");
	}

	void ImageProcessingCpuNode::configurationEntryChanged(const std::string& configKey)
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
	}

	template <typename InputType, typename OutputType>
	std::shared_ptr<ContainerBase> ImageProcessingCpuNode::processTemplated(std::shared_ptr<const Container<InputType> > imageData, vec3s size)
	{
		// here the actual processing happens!

		size_t width = size.x;
		size_t height = size.y;
		size_t depth = size.z;

		// make sure the data is in cpu memory
		auto inImageData = imageData;
		if (!inImageData->isHost() && !inImageData->isBoth())
		{
			inImageData = make_shared<Container<InputType> >(LocationHost, *inImageData);
		}
		// Get pointer to the actual memory block
		const InputType* pInputImage = inImageData->get();

		// prepare the output memory
		auto outImageData = make_shared<Container<OutputType> >(LocationHost, inImageData->getStream(), width*height*depth);
		// Get pointer to the actual memory block
		OutputType* pOutputImage = outImageData->get();

		// process the image
		for (size_t z = 0; z < depth; z++)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t x = 0; x < width; x++)
				{
					// Perform a pixel-wise operation on the image

					// Get the input pixel value and cast it to out working type.
					// As this should in general be a type with wider range / precision, this cast does not loose anything.
					ImageProcessingCpuNode::WorkType inPixel = pInputImage[x + y*width + z *width*height];

					// Perform operation, in this case multiplication
					WorkType value = inPixel * static_cast<WorkType>(m_factor);

					// Store the output pixel value.
					// Because this is templated, we need to cast from "WorkType" to "OutputType".
					// This should happen in a sane way, that is with clamping. There is a helper for that!
					pOutputImage[x + y*width + z *width*height] = clampCast<OutputType>(value);
				}
			}
		}

		// return the result!
		return outImageData;
	}

	template <typename InputType>
	std::shared_ptr<ContainerBase> ImageProcessingCpuNode::processTemplateSelection(std::shared_ptr<const Container<InputType> > imageData, vec3s size)
	{
		// With the function already templated on the input type, handle the desired output type.
		switch (m_outputType)
		{
		case supra::TypeUint8:
			return processTemplated<InputType, uint8_t>(imageData, size);
			break;
		case supra::TypeInt16:
			return processTemplated<InputType, int16_t>(imageData, size);
			break;
		case supra::TypeFloat:
			return processTemplated<InputType, float>(imageData, size);
			break;
		default:
			logging::log_error("ImageProcessingCpuNode: Output image type not supported");
			break;
		}
		return nullptr;
	}

	shared_ptr<RecordObject> ImageProcessingCpuNode::checkTypeAndProcess(shared_ptr<RecordObject> inObj)
	{
		shared_ptr<USImage> pImage = nullptr;
		if (inObj && inObj->getType() == TypeUSImage)
		{
			shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
			if (pInImage)
			{
				// lock the object mutex to make sure no parameters are changed during processing
				unique_lock<mutex> l(m_mutex);
				m_callFrequency.measure();

				std::shared_ptr<ContainerBase> pImageProcessedData;

				// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
				// This first switch handles the different input data types. There is no need to support all types, 
				// only those meaningful for the operation of the node.
				switch (pInImage->getDataType())
				{
				case TypeUint8:
					pImageProcessedData = processTemplateSelection<uint8_t>(pInImage->getData<uint8_t>(), pInImage->getSize());
					break;
				case TypeInt16:
					pImageProcessedData = processTemplateSelection<int16_t>(pInImage->getData<int16_t>(), pInImage->getSize());
					break;
				case TypeFloat:
					pImageProcessedData = processTemplateSelection<float>(pInImage->getData<float>(), pInImage->getSize());
					break;
				default:
					logging::log_error("ImageProcessingCpuNode: Input image type not supported");
					break;
				}
				m_callFrequency.measureEnd();

				// Wrap the returned Container in an USImage with the same size etc.
				pImage = make_shared<USImage>(
					pInImage->getSize(),
					pImageProcessedData,
					pInImage->getImageProperties(),
					pInImage->getReceiveTimestamp(),
					pInImage->getSyncTimestamp());
			}
			else {
				logging::log_error("ImageProcessingCpuNode: could not cast object to USImage type, is it in suppored ElementType?");
			}
		}
		return pImage;
	}
}