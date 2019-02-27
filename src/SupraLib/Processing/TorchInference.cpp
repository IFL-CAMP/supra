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

#include "TorchInference.h"

using namespace std;

namespace supra
{
	TorchInference::TorchInference(const std::string& modelFilename)
		: m_modelFilename{ modelFilename }
		, m_torchModule{ nullptr }
	{
		loadModule();
	}

	//shared_ptr<RecordObject> TorchInference::checkTypeAndProcess(shared_ptr<RecordObject> inObj)
	//{
	//	shared_ptr<RecordObject> pOut = nullptr;
	//	if (inObj && inObj->getType() == TypeUSImage)
	//	{
	//		shared_ptr<USImage> pInImage = dynamic_pointer_cast<USImage>(inObj);
	//		if (pInImage)
	//		{
	//			// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
	//			// This first switch handles the different input data types. There is no need to support all types, 
	//			// only those meaningful for the operation of the node.
	//			switch (pInImage->getDataType())
	//			{
	//			case TypeUint8:
	//				pOut = processTemplateSelection<uint8_t>(pInImage->getData<uint8_t>(), pInImage->getSize(), 1, pInImage->getImageProperties());
	//				break;
	//			case TypeInt16:
	//				pOut = processTemplateSelection<int16_t>(pInImage->getData<int16_t>(), pInImage->getSize(), 1, pInImage->getImageProperties());
	//				break;
	//			case TypeFloat:
	//				pOut = processTemplateSelection<float>(pInImage->getData<float>(), pInImage->getSize(), 1, pInImage->getImageProperties());
	//				break;
	//			default:
	//				logging::log_error("TorchInference: Input image type not supported");
	//				break;
	//			}
	//		}
	//		else {
	//			logging::log_error("TorchInference: could not cast object to USImage type, although its type is TypeUSImage.");
	//		}
	//	}
	//	else if (inObj && inObj->getType() == TypeUSRawData)
	//	{
	//		shared_ptr<USRawData> pInRawData = dynamic_pointer_cast<USRawData>(inObj);
	//		if (pInRawData)
	//		{
	//			// The input and output types have to be determined dynamically. We do this in to stages of templated functions.
	//			// This first switch handles the different input data types. There is no need to support all types,
	//			// only those meaningful for the operation of the node.
	//			vec3s size{pInRawData->getNumSamples(), pInRawData->getNumReceivedChannels(), pInRawData->getNumScanlines()};
	//			switch (pInRawData->getDataType())
	//			{
	//				case TypeUint8:
	//					pOut = processTemplateSelection<uint8_t>(pInRawData->getData<uint8_t>(), size, 0, pInRawData->getImageProperties());
	//					break;
	//				case TypeInt16:
	//					pOut = processTemplateSelection<int16_t>(pInRawData->getData<int16_t>(), size, 0, pInRawData->getImageProperties());
	//					break;
	//				case TypeFloat:
	//					pOut = processTemplateSelection<float>(pInRawData->getData<float>(), size, 0, pInRawData->getImageProperties());
	//					break;
	//				default:
	//					logging::log_error("TorchInference: Input image type not supported");
	//					break;
	//			}
	//		}
	//		else {
	//			logging::log_error("TorchInference: could not cast object to USRawData type, although its type is TypeUSRawData.");
	//		}
	//	}
	//	return pOut;
	//}

	void TorchInference::loadModule() {
		m_torchModule = nullptr;
		if (m_modelFilename != "")
		{
			try {
				std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(m_modelFilename);
				module->to(torch::kCUDA);
				m_torchModule = module;
			}
			catch (c10::Error e)
			{
				logging::log_error("TorchInference: Exception (c10::Error) while loading model '", m_modelFilename, "'");
				logging::log_error("TorchInference: ", e.what());
				logging::log_error("TorchInference: ", e.msg_stack());
				m_torchModule = nullptr;
			}
			catch (std::runtime_error e)
			{
				logging::log_error("TorchInference: Exception (std::runtime_error) while loading model '", m_modelFilename, "'");
				logging::log_error("TorchInference: ", e.what());
				m_torchModule = nullptr;
			}
		}
	}

	at::Tensor supra::TorchInference::convertDataType(at::Tensor tensor, DataType datatype)
	{
		switch (datatype)
		{
		case TypeInt8:
			tensor = tensor.to(caffe2::TypeMeta::Make<int8_t>());
			break;
		case TypeUint8:
			tensor = tensor.to(caffe2::TypeMeta::Make<uint8_t>());
			break;
		case TypeInt16:
			tensor = tensor.to(caffe2::TypeMeta::Make<int16_t>());
			break;
		case TypeUint16:
			tensor = tensor.to(caffe2::TypeMeta::Make<uint16_t>());
			break;
		case TypeInt32:
			tensor = tensor.to(caffe2::TypeMeta::Make<int32_t>());
			break;
		/*case TypeUint32:
			tensor = tensor.to(caffe2::TypeMeta::Make<uint32_t>());
			break;*/
		case TypeInt64:
			tensor = tensor.to(caffe2::TypeMeta::Make<int64_t>());
			break;
		/*case TypeUint64:
			tensor = tensor.to(caffe2::TypeMeta::Make<uint64_t>());
			break;*/
		case TypeFloat:
			tensor = tensor.to(caffe2::TypeMeta::Make<float>());
			break;
		case TypeDouble:
			tensor = tensor.to(caffe2::TypeMeta::Make<double>());
			break;
		default:
			logging::log_error("TorchInference: convertDataType: Type '", datatype, "' is not supported.");
			break;
		}
		return tensor;
	}

	at::Tensor supra::TorchInference::changeLayout(at::Tensor tensor, const std::string & currentLayout, const std::string & outLayout)
	{
		if (currentLayout != outLayout)
		{
			int inDimensionC = (currentLayout[0] == 'C' ? 1 : (currentLayout[2] == 'C' ? 2 : 3));
			int inDimensionW = (currentLayout[0] == 'W' ? 1 : (currentLayout[2] == 'W' ? 2 : 3));
			int inDimensionH = (currentLayout[0] == 'H' ? 1 : (currentLayout[2] == 'H' ? 2 : 3));
			int outDimensionC = (outLayout[0] == 'C' ? 1 : (outLayout[2] == 'C' ? 2 : 3));
			int outDimensionW = (outLayout[0] == 'W' ? 1 : (outLayout[2] == 'W' ? 2 : 3));
			int outDimensionH = (outLayout[0] == 'H' ? 1 : (outLayout[2] == 'H' ? 2 : 3));
			
			std::vector<int64_t> permutation(4, 0);
			permutation[0] = 0;
			permutation[outDimensionC] = inDimensionC;
			permutation[outDimensionW] = inDimensionW;
			permutation[outDimensionH] = inDimensionH;
			tensor = tensor.permute(permutation);
		}
		return tensor;
	}
}