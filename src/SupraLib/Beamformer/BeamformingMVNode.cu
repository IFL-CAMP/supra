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

#include "BeamformingMVNode.h"

#include "USImage.h"
#include "USRawData.h"

#include <utilities/Logging.h>
#include <utilities/cudaUtility.h>
//#include <algorithm>
using namespace std;

namespace supra
{
	/// Verifies a cuda call returned "CUBLAS_STATUS_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise.
	#define cublasSafeCall(_err_) cublasSafeCall2(_err_, __FILE__, __LINE__, FUNCNAME_PORTABLE)

	/// Verifies a cuda call returned "CUBLAS_STATUS_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise. Calles by cudaSafeCall
	inline bool cublasSafeCall2(cublasStatus_t err, const char* file, int line, const char* func) {

		//#ifdef CUDA_ERROR_CHECK
		if (CUBLAS_STATUS_SUCCESS != err) {
			char buf[1024];
			sprintf(buf, "CUBLAS Error (in \"%s\", Line: %d, %s): %d\n", file, line, func, err);
			printf("%s", buf);
			logging::log_error(buf);
			return false;
		}

		//#endif
		return true;
	}

	template <typename ChannelDataType>
	__global__ void computeRmatrices(const ChannelDataType* rawData,
		uint32_t numSamples, uint32_t numChannels, uint32_t scanlineIdx,
		uint32_t subArraySize, float* Rmatrices)
	{
		int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
		int sampleIdx = (blockIdx.y * gridDim.x) + blockIdx.x;
		
		if (sampleIdx < numSamples)
		{
			int numSubArrays = numChannels - subArraySize + 1;
			int numelR = subArraySize*subArraySize;
			float* R = &Rmatrices[sampleIdx * numelR];

			for (int subArray = 0; subArray < numSubArrays; subArray++)
			{
				for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
				{
					int colIdx = matrixIdx % subArraySize;
					int rowIdx = matrixIdx / subArraySize;

					float xCol = rawData[sampleIdx + (subArray + colIdx)*numSamples + scanlineIdx*numChannels*numSamples];
					float xRow = rawData[sampleIdx + (subArray + rowIdx)*numSamples + scanlineIdx*numChannels*numSamples];

					atomicAdd(&R[matrixIdx], xCol*xRow);
				}
			}
		}
	}

	__global__ void computeTemporalSmoothRmatrices(const float* Rmatrices,
		uint32_t numSamples, uint32_t subArraySize, uint32_t numSubArrays, 
		uint32_t temporalSmoothing, float* TempRmatrices)
	{
		int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
		int sampleIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

		if (sampleIdx < numSamples)
		{
			int numelR = subArraySize*subArraySize;

			int firstIdx = max(0, sampleIdx - temporalSmoothing);
			int lastIdx = min(numSamples - 1, sampleIdx + temporalSmoothing);

			float scaling = 1 / ((lastIdx - firstIdx + 1)*(numSubArrays));
			for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
			{
				float finalEntry = 0.0f;
				for (int tempIdx = firstIdx; tempIdx <= lastIdx; tempIdx++)
				{
					finalEntry += Rmatrices[matrixIdx + tempIdx*numelR];
				}
				//TempRmatrices[matrixIdx + sampleIdx*numelR] = finalEntry*scaling;
				TempRmatrices[0] = finalEntry*scaling;
			}
		}
	}

	__inline__ __device__
	int warpAllReduceSum(int val) {
		for (int mask = warpSize / 2; mask > 0; mask /= 2)
		{
			val += __shfl_xor(val, mask);
		}
		return val;
	}

	__global__ void addDiagonalLoading(float* Rmatrices,
		uint32_t numSamples, uint32_t subArraySize)
	{
		int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
		int sampleIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

		if (sampleIdx < numSamples)
		{
			int numelR = subArraySize*subArraySize;

			float* R = &Rmatrices[sampleIdx*numelR];

			// compute trace in one block
			float localSum = 0.0f;
			for (int diagIdx = tIdx; diagIdx < subArraySize; diagIdx += blockDim.x*blockDim.y)
			{
				int matrixIdx = diagIdx * (subArraySize + 1);
				
				localSum += R[matrixIdx];
			}
			float trace = warpAllReduceSum(localSum);
			float loading = (1.0f / static_cast<float>(subArraySize)) * trace;

			for (int diagIdx = tIdx; diagIdx < subArraySize; diagIdx += blockDim.x*blockDim.y)
			{
				int matrixIdx = diagIdx * (subArraySize + 1);

				R[matrixIdx] += loading;
			}
		}
	}

	template <typename ChannelDataType, typename ImageDataType>
	__global__ void applyWeights(
		const float* RinverseA, 
		const float* A, 
		const ChannelDataType* rawData,
		uint32_t numSamples, 
		uint32_t numChannels, 
		uint32_t numScanlines, 
		uint32_t scanlineIdx, 
		uint32_t subArraySize, 
		ImageDataType* beamformed)
	{
		int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
		int sampleIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

		if (sampleIdx < numSamples)
		{
			int numSubArrays = numChannels - subArraySize + 1;

			// compute weight scaling <a, R\a>
			const float* RinvAloc = &RinverseA[sampleIdx * subArraySize];
			const float* Aloc = &A[sampleIdx * subArraySize];
			float weightScaling = 0.0f;
			for (int vectorIdx = tIdx; vectorIdx < subArraySize; vectorIdx += blockDim.x*blockDim.y)
			{
				weightScaling += RinvAloc[vectorIdx] * Aloc[vectorIdx];
			}
			weightScaling = 1 / (warpAllReduceSum(weightScaling) * numSubArrays);

			// compute one sample at a time, according to spatial smoothing
			float beamformedSample = 0.0f;
			for (int vectorIdx = tIdx; vectorIdx < subArraySize; vectorIdx += blockDim.x*blockDim.y)
			{
				float sample = 0.0;
				for (int subArray = 0; subArray < numSubArrays; subArray++)
				{
					sample += rawData[sampleIdx + (subArray + vectorIdx)*numSamples + scanlineIdx*numChannels*numSamples];
				}
				beamformedSample += sample*RinvAloc[vectorIdx] * weightScaling;
			}
			beamformedSample = warpAllReduceSum(beamformedSample);
			if (tIdx == 0)
			{
				beamformed[scanlineIdx + sampleIdx * numScanlines] = beamformedSample;
			}
		}
	}

	// perform the receive beamforming
	template <typename ChannelDataType, typename ImageDataType>
	shared_ptr<USImage<ImageDataType> > performRxBeamforming(
		shared_ptr<const USRawData<ChannelDataType> > rawData,
		uint32_t subArraySize,
		uint32_t temporalSmoothing,
		cublasHandle_t cublasH)
	{
		//Ensure the raw-data are on the gpu
		auto gRawData = rawData->getData();
		if (!rawData->getData()->isGPU() && !rawData->getData()->isBoth())
		{
			gRawData = std::make_shared<Container<ChannelDataType> >(LocationGpu, *gRawData);
		}
		auto stream = gRawData->getStream();

		uint32_t numScanlines = static_cast<uint32_t>(rawData->getNumScanlines());
		uint32_t numSamples   = static_cast<uint32_t>(rawData->getNumSamples());
		uint32_t numChannels  = static_cast<uint32_t>(rawData->getNumReceivedChannels());
		if (subArraySize == 0)
		{
			subArraySize = numChannels;
		}

		uint32_t numSubArrays = numChannels - subArraySize + 1;

		size_t numelOut = numScanlines*numSamples;
		shared_ptr<Container<ImageDataType> > pData = std::make_shared<Container<ImageDataType> >(ContainerLocation::LocationGpu, stream, numelOut);

		size_t numelRmatrices = subArraySize*subArraySize* numSamples;
		shared_ptr<Container<float> > Rmatrices = std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, numelRmatrices);
		shared_ptr<Container<float> > RmatricesTempSmooth = std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, numelRmatrices);
		shared_ptr<Container<float> > Avectors = std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, std::vector<float>(subArraySize*numSamples, 1.0f));
		shared_ptr<Container<float> > AvectorsOrg = std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, std::vector<float>(subArraySize*numSamples, 1.0f));

		shared_ptr<Container<int> > pivotizationArray = std::make_shared<Container<int> >(ContainerLocation::LocationGpu, stream, subArraySize* numSamples);
		std::vector<int> cublasInfoArray(numSamples);

		int numelR = subArraySize*subArraySize;
		std::vector<float*> Rpointers(numSamples);
		std::vector<float*> Apointers(numSamples);
		for (uint32_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
		{
			Rpointers[sampleIdx] = RmatricesTempSmooth->get() + sampleIdx* numelR;
			Apointers[sampleIdx] = Avectors->get() + sampleIdx* subArraySize;
		}
		shared_ptr<Container<float*> > RpointersDevice = std::make_shared<Container<float*> >(ContainerLocation::LocationGpu, stream, Rpointers);
		shared_ptr<Container<float*> > ApointersDevice = std::make_shared<Container<float*> >(ContainerLocation::LocationGpu, stream, Apointers);

		for (uint32_t scanlineIdx = 0; scanlineIdx < numScanlines; scanlineIdx++)
		{
			cudaSafeCall(cudaMemsetAsync(Rmatrices->get(), 0, numelRmatrices * sizeof(float), stream));
			cudaSafeCall(cudaMemcpyAsync(Avectors->get(), AvectorsOrg->get(), subArraySize*numSamples * sizeof(float), cudaMemcpyDefault, stream));
			
			//TEST
			cudaSafeCall(cudaDeviceSynchronize());

			dim3 blockSize(32, 1);
			dim3 gridSize(numSamples, 1);
			computeRmatrices<<<gridSize, blockSize, 0, stream>>>(
				gRawData->get(),
				numSamples,
				numChannels,
				scanlineIdx,
				subArraySize,
				Rmatrices->get()
			);
			cudaSafeCall(cudaPeekAtLastError());
			//TEST
			cudaSafeCall(cudaDeviceSynchronize());

			computeTemporalSmoothRmatrices<<<gridSize, blockSize, 0, stream>>>(
				Rmatrices->get(),
				numSamples,
				subArraySize,
				numSubArrays,
				temporalSmoothing,
				Rmatrices->get()//RmatricesTempSmooth->get()
			);
			cudaSafeCall(cudaPeekAtLastError());
			//TEST
			cudaSafeCall(cudaDeviceSynchronize());

			/*
			addDiagonalLoading<<<gridSize, dim3(32, 1), 0, stream>>>(
				RmatricesTempSmooth->get(),
				numSamples, subArraySize
			);
			cudaSafeCall(cudaPeekAtLastError());
			//TEST
			cudaSafeCall(cudaDeviceSynchronize());

			cublasSafeCall(cublasSgetrfBatched(
				cublasH,
				subArraySize,
				(float**)RpointersDevice->get(),
				subArraySize,
				pivotizationArray->get(),
				cublasInfoArray.data(),
				numSamples));
			//TEST
			cudaSafeCall(cudaDeviceSynchronize());*/

			/*cublasSafeCall(cublasSgetrsBatched(
				cublasH,
				CUBLAS_OP_N,
				subArraySize,
				1,
				(const float**)RpointersDevice->get(),
				subArraySize,
				pivotizationArray->get(),
				(float**)ApointersDevice->get(),
				subArraySize,
				cublasInfoArray.data(),
				numSamples));

			// calculate beamforming weights from that and perform beamforming
			applyWeights<<<gridSize, dim3(32, 1), 0, stream>>>(
				Avectors->get(),
				AvectorsOrg->get(),
				gRawData->get(),
				numSamples,
				numChannels,
				numScanlines,
				scanlineIdx,
				subArraySize,
				pData->get()
			);
			cudaSafeCall(cudaPeekAtLastError());*/
		}

		auto retImage = std::make_shared<USImage<ImageDataType> >(
			vec2s{ numScanlines, numSamples },
			pData,
			rawData->getImageProperties(),
			rawData->getReceiveTimestamp(),
			rawData->getSyncTimestamp());

		return retImage;
	}

	BeamformingMVNode::BeamformingMVNode(tbb::flow::graph & graph, const std::string & nodeID)
		: AbstractNode(nodeID)
		, m_node(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndBeamform(inObj); })
		, m_lastSeenImageProperties(nullptr)
	{
		m_callFrequency.setName("BeamformingMV");
		m_valueRangeDictionary.set<uint32_t>("subArraySize", 0, 64, 0, "Sub-array size");
		m_valueRangeDictionary.set<uint32_t>("temporalSmoothing", 0, 10, 3, "temporal smoothing");
		
		configurationChanged();

		cublasSafeCall(cublasCreate(&m_cublasH));
		cublasSafeCall(cublasSetAtomicsMode(m_cublasH, CUBLAS_ATOMICS_ALLOWED));
	}

	BeamformingMVNode::~BeamformingMVNode()
	{
		cublasSafeCall(cublasDestroy(m_cublasH));
	}

	void BeamformingMVNode::configurationChanged()
	{
		m_subArraySize = m_configurationDictionary.get<uint32_t>("subArraySize");
		m_temporalSmoothing = m_configurationDictionary.get<uint32_t>("temporalSmoothing");
	}

	void BeamformingMVNode::configurationEntryChanged(const std::string& configKey)
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
		if (m_lastSeenImageProperties)
		{
			updateImageProperties(m_lastSeenImageProperties);
		}
	}

	shared_ptr<RecordObject> BeamformingMVNode::checkTypeAndBeamform(shared_ptr<RecordObject> inObj)
	{
		unique_lock<mutex> l(m_mutex);

		shared_ptr<USImage<int16_t> > pImageRF = nullptr;
		if (inObj->getType() == TypeUSRawData)
		{
			shared_ptr<const USRawData<int16_t> > pRawData = dynamic_pointer_cast<const USRawData<int16_t>>(inObj);
			if (pRawData)
			{
				if (pRawData->getImageProperties()->getImageState() == USImageProperties::RawDelayed)
				{
					m_callFrequency.measure();

					cublasSafeCall(cublasSetStream(m_cublasH, pRawData->getData()->getStream()));
					pImageRF = performRxBeamforming<int16_t, int16_t>(
						pRawData, m_subArraySize, m_temporalSmoothing, m_cublasH);
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

	void BeamformingMVNode::updateImageProperties(std::shared_ptr<const USImageProperties> imageProperties)
	{
		m_lastSeenImageProperties = imageProperties;
		m_editedImageProperties = make_shared<USImageProperties>(*imageProperties);
		m_editedImageProperties->setImageState(USImageProperties::RF);
		m_editedImageProperties->setSpecificParameter("BeamformingMVNode.subArraySize", m_subArraySize);
		m_editedImageProperties->setSpecificParameter("BeamformingMVNode.temporalSmoothing", m_temporalSmoothing);
	}
}