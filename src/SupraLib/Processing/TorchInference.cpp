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

#include <torch/jit.h>

using namespace std;

namespace supra
{
	TorchInference::TorchInference(
		const std::string& modelFilename,
		const std::string& inputNormalization,
		const std::string& outputDenormalization)
		: m_modelFilename{ modelFilename }
		, m_inputNormalization{inputNormalization}
		, m_outputDenormalization{outputDenormalization}
		, m_torchModule{ nullptr }
		, m_inputNormalizationModule{ nullptr }
		, m_outputDenormalizationModule{ nullptr }
	{
		if (m_inputNormalization == "")
		{
			m_inputNormalization = "a";
		}
		if (m_outputDenormalization == "")
		{
			m_outputDenormalization = "a";
		}

		loadModule();
	}

	void TorchInference::loadModule() {
		m_torchModule = nullptr;
		if (m_modelFilename != "" && m_inputNormalization != "" && m_outputDenormalization != "" )
		{
			try {
				std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(m_modelFilename);
				module->to(torch::kCUDA);
				m_torchModule = module;

				m_inputNormalizationModule = torch::jit::compile(
					"  def normalize(a):\n    return " + m_inputNormalization + "\n");
				m_inputNormalizationModule->to(torch::kCUDA);
				m_outputDenormalizationModule = torch::jit::compile(
					"  def denormalize(a):\n    return " + m_outputDenormalization + "\n");
				m_outputDenormalizationModule->to(torch::kCUDA);
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
		case TypeInt64:
			tensor = tensor.to(caffe2::TypeMeta::Make<int64_t>());
			break;
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