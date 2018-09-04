// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2017, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "RxBeamformerMV.h"

#include "USImage.h"
#include "USRawData.h"

#include <utilities/Logging.h>
#include <utilities/cudaUtility.h>
#include <utilities/cublasUtility.h>

using namespace std;

namespace supra
{
	template <typename ChannelDataType>
	__global__ void computeSubArrayMasks(
		const ChannelDataType* rawData,
		uint32_t numSamples, uint32_t numChannels,
		uint32_t scanlineIdx, uint32_t sampleIdxStart,
		uint32_t subArraySize, uint8_t* subArrayMasks)
	{
		int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
		int sampleIdxLocal = (blockIdx.y * gridDim.x) + blockIdx.x;
		int sampleIdx = sampleIdxLocal + sampleIdxStart;
		
		if (sampleIdx < numSamples)
		{
			int numSubArrays = numChannels - subArraySize + 1;

			for (int subArray = tIdx; subArray < numSubArrays; subArray++)
			{
				ChannelDataType xLeft = rawData[sampleIdx + (subArray + 0)*numSamples + scanlineIdx*numChannels*numSamples];
				ChannelDataType xRight = rawData[sampleIdx + (subArray + subArraySize - 1)*numSamples + scanlineIdx*numChannels*numSamples];
				if (xLeft != 0 && xRight != 0)
				{
					subArrayMasks[subArray + sampleIdxLocal*numSubArrays] = 1;
				}
			}
		}
	}

	template <typename ChannelDataType>
	__global__ void computeRmatrices(const ChannelDataType* rawData,
		uint32_t numSamples, uint32_t numChannels, uint32_t scanlineIdx, uint32_t sampleIdxStart,
		uint32_t subArraySize, const uint8_t * subArrayMasks, float* Rmatrices)
	{
		int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
		int sampleIdxLocal = (blockIdx.y * gridDim.x) + blockIdx.x;
		int sampleIdx = sampleIdxLocal + sampleIdxStart;
		
		if (sampleIdx < numSamples)
		{
			int numSubArrays = numChannels - subArraySize + 1;
			int numelR = subArraySize*subArraySize;
			float* R = &Rmatrices[sampleIdxLocal * numelR];

			for (int subArray = 0; subArray < numSubArrays; subArray++)
			{
				if (subArrayMasks[subArray + sampleIdxLocal*numSubArrays] != 0)
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

			int firstIdx = max(0, sampleIdx - (int)(temporalSmoothing));
			int lastIdx = min((int)(numSamples) - 1, sampleIdx + (int)(temporalSmoothing));

			float scaling = 1.0f / (static_cast<float>(lastIdx - firstIdx + 1)*(numSubArrays));
			for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
			{
				float finalEntry = 0.0f;
				for (int tempIdx = firstIdx; tempIdx <= lastIdx; tempIdx++)
				{
					finalEntry += Rmatrices[matrixIdx + tempIdx*numelR];
				}
				TempRmatrices[matrixIdx + sampleIdx*numelR] = finalEntry*scaling;
			}
		}
	}

	template <typename T>
	__inline__ __device__ T warpAllReduceSum(T val) {
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
		uint32_t sampleIdxStart,
		uint32_t subArraySize,
		const uint8_t * subArrayMasks,
		ImageDataType* beamformed)
	{
		int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
		int sampleIdxLocal = (blockIdx.y * gridDim.x) + blockIdx.x;
		int sampleIdx = sampleIdxLocal + sampleIdxStart;

		if (sampleIdx < numSamples)
		{
			int numSubArrays = numChannels - subArraySize + 1;

			// compute weight scaling <a, R\a>
			const float* RinvAloc = &RinverseA[sampleIdxLocal * subArraySize];
			const float* Aloc = &A[sampleIdxLocal * subArraySize];
			float weightScaling = 0.0f;
			for (int vectorIdx = tIdx; vectorIdx < subArraySize; vectorIdx += blockDim.x*blockDim.y)
			{
				weightScaling += RinvAloc[vectorIdx] * Aloc[vectorIdx];
			}
			weightScaling = 1.0f / (warpAllReduceSum(weightScaling) * numSubArrays);
			
			// compute one sample at a time, according to spatial smoothing
			float beamformedSample = 0.0f;
			for (int vectorIdx = tIdx; vectorIdx < subArraySize; vectorIdx += blockDim.x*blockDim.y)
			{
				float sample = 0.0;
				for (int subArray = 0; subArray < numSubArrays; subArray++)
				{
					if (subArrayMasks[subArray + sampleIdxLocal*numSubArrays] != 0)
					{
						sample += rawData[sampleIdx + (subArray + vectorIdx)*numSamples + scanlineIdx*numChannels*numSamples];
					}
				}
				beamformedSample += sample * RinvAloc[vectorIdx] * weightScaling;
			}
			beamformedSample = warpAllReduceSum(beamformedSample);
			if (tIdx == 0)
			{
				if (abs(beamformedSample) > 1e7 || ::isnan(beamformedSample))
				{
					beamformedSample = 0.0f;
				}
				beamformed[scanlineIdx + sampleIdx * numScanlines] =
					clampCast<ImageDataType>(beamformedSample * numChannels);
			}
		}
	}

	// perform the receive beamforming
	template <typename ChannelDataType, typename ImageDataType>
	shared_ptr<USImage> performRxBeamforming(
		shared_ptr<const USRawData> rawData,
		uint32_t subArraySize,
		uint32_t temporalSmoothing,
		cublasHandle_t cublasH)
	{
		int sampleBlockSize = 2000;//128;

		//Ensure the raw-data are on the gpu
		auto gRawData = rawData->getData<ChannelDataType>();
		if (!gRawData->isGPU() && !gRawData->isBoth())
		{
			gRawData = std::make_shared<Container<ChannelDataType> >(LocationGpu, *gRawData);
		}
		auto stream = gRawData->getStream();

		uint32_t numScanlines = static_cast<uint32_t>(rawData->getNumScanlines());
		uint32_t numSamples   = static_cast<uint32_t>(rawData->getNumSamples());
		uint32_t numChannels  = static_cast<uint32_t>(rawData->getNumReceivedChannels());
		if (subArraySize == 0)
		{
			subArraySize = numChannels/2;
		}

		uint32_t numSubArrays = numChannels - subArraySize + 1;

		size_t numelOut = numScanlines*numSamples;
		shared_ptr<Container<ImageDataType> > pData = std::make_shared<Container<ImageDataType> >(ContainerLocation::LocationGpu, stream, numelOut);

		size_t numelRmatrices = subArraySize*subArraySize* sampleBlockSize;
		shared_ptr<Container<float> > Rmatrices = 
			std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, numelRmatrices);
		shared_ptr<Container<float> > RmatricesTempSmooth =
			std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, numelRmatrices);
		shared_ptr<Container<float> > Avectors = 
			std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, std::vector<float>(subArraySize*sampleBlockSize, 1.0f));
		shared_ptr<Container<float> > AvectorsOrg = 
			std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, std::vector<float>(subArraySize*sampleBlockSize, 1.0f));
		shared_ptr<Container<uint8_t> > subArrayMask = 
			std::make_shared<Container<uint8_t> >(ContainerLocation::LocationGpu, stream, numSubArrays*sampleBlockSize);

		shared_ptr<Container<int> > pivotizationArray = 
			std::make_shared<Container<int> >(ContainerLocation::LocationGpu, stream, subArraySize* sampleBlockSize);
		std::vector<int> cublasInfoArrayHost(sampleBlockSize);
		shared_ptr<Container<int> > cublasInfoArrayDevice = std::make_shared<Container<int> >(ContainerLocation::LocationGpu, stream, sampleBlockSize);

		int numelR = subArraySize*subArraySize;
		std::vector<float*> Rpointers(sampleBlockSize);
		std::vector<float*> Apointers(sampleBlockSize);
		for (uint32_t sampleIdx = 0; sampleIdx < sampleBlockSize; sampleIdx++)
		{
			Rpointers[sampleIdx] = RmatricesTempSmooth->get() + sampleIdx* numelR;
			Apointers[sampleIdx] = Avectors->get() + sampleIdx* subArraySize;
		}
		shared_ptr<Container<float*> > RpointersDevice = std::make_shared<Container<float*> >(ContainerLocation::LocationGpu, stream, Rpointers);
		shared_ptr<Container<float*> > ApointersDevice = std::make_shared<Container<float*> >(ContainerLocation::LocationGpu, stream, Apointers);

		for (uint32_t scanlineIdx = 0; scanlineIdx < numScanlines; scanlineIdx++)
		{
			for (uint32_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx += sampleBlockSize)
			{
				uint32_t numSamplesBatch = min(sampleBlockSize, numSamples - sampleIdx);

				cudaSafeCall(cudaMemsetAsync(Rmatrices->get(), 0, numelRmatrices * sizeof(float), stream));
				cudaSafeCall(cudaMemcpyAsync(Avectors->get(), AvectorsOrg->get(), subArraySize*sampleBlockSize * sizeof(float), cudaMemcpyDefault, stream));
				cudaSafeCall(cudaMemsetAsync(subArrayMask->get(), 0, numSubArrays*sampleBlockSize * sizeof(uint8_t), stream));

				//TEST
				cudaSafeCall(cudaDeviceSynchronize());

				dim3 blockSize(32, 1);
				dim3 gridSize(numSamplesBatch, 1);

				computeSubArrayMasks<<<gridSize, blockSize, 0, stream>>>(
					gRawData->get(),
					numSamples, 
					numChannels,
					scanlineIdx, 
					sampleIdx,
					subArraySize,
					subArrayMask->get());
				//TEST
				cudaSafeCall(cudaDeviceSynchronize());
					
				computeRmatrices<<<gridSize, blockSize, 0, stream>>> (
					gRawData->get(),
					numSamples,
					numChannels,
					scanlineIdx,
					sampleIdx,
					subArraySize,
					subArrayMask->get(),
					Rmatrices->get()
					);
				cudaSafeCall(cudaPeekAtLastError());
				//TEST
				cudaSafeCall(cudaDeviceSynchronize());

				//TEST
				//auto RmatHost1 = make_shared<Container<float> >(ContainerLocation::LocationHost, *Rmatrices);

				computeTemporalSmoothRmatrices<<<gridSize, blockSize, 0, stream>>> (
					Rmatrices->get(),
					numSamplesBatch,
					subArraySize,
					numSubArrays,
					temporalSmoothing,
					RmatricesTempSmooth->get()
					);
				cudaSafeCall(cudaPeekAtLastError());
				//TEST
				cudaSafeCall(cudaDeviceSynchronize());

				//TEST
				//auto RmatHost2 = make_shared<Container<float> >(ContainerLocation::LocationHost, *Rmatrices);

				addDiagonalLoading<<<gridSize, dim3(32, 1), 0, stream>>> (
					RmatricesTempSmooth->get(),
					numSamplesBatch, subArraySize
					);
				cudaSafeCall(cudaPeekAtLastError());
				//TEST
				cudaSafeCall(cudaDeviceSynchronize());

				//TEST
				//auto RmatHost3 = make_shared<Container<float> >(ContainerLocation::LocationHost, *RmatricesTempSmooth);

				cublasSafeCall(cublasSgetrfBatched(
					cublasH,
					subArraySize,
					(float**)RpointersDevice->get(),
					subArraySize,
					pivotizationArray->get(),
					cublasInfoArrayDevice->get(),
					numSamplesBatch));
				//TEST
				cudaSafeCall(cudaDeviceSynchronize());


				cublasSafeCall(cublasSgetrsBatched(
					cublasH,
					CUBLAS_OP_N,
					subArraySize,
					1,
					(const float**)RpointersDevice->get(),
					subArraySize,
					pivotizationArray->get(),
					(float**)ApointersDevice->get(),
					subArraySize,
					cublasInfoArrayHost.data(),
					numSamplesBatch));
				//TEST
				cudaSafeCall(cudaDeviceSynchronize());

				// calculate beamforming weights from that and perform beamforming
				applyWeights<<<gridSize, dim3(32, 1), 0, stream>>> (
					Avectors->get(),
					AvectorsOrg->get(),
					gRawData->get(),
					numSamples,
					numChannels,
					numScanlines,
					scanlineIdx,
					sampleIdx,
					subArraySize,
					subArrayMask->get(),
					pData->get()
					);
				cudaSafeCall(cudaPeekAtLastError());
				//TEST
				cudaSafeCall(cudaDeviceSynchronize());
			}
		}

		auto retImage = std::make_shared<USImage>(
			vec2s{ numScanlines, numSamples },
			pData,
			rawData->getImageProperties(),
			rawData->getReceiveTimestamp(),
			rawData->getSyncTimestamp());

		return retImage;
	}

	template 
	shared_ptr<USImage> performRxBeamforming<int16_t, int16_t>(
		shared_ptr<const USRawData> rawData,
		uint32_t subArraySize,
		uint32_t temporalSmoothing,
		cublasHandle_t cublasH);
	template
		shared_ptr<USImage> performRxBeamforming<int16_t, float>(
			shared_ptr<const USRawData> rawData,
			uint32_t subArraySize,
			uint32_t temporalSmoothing,
			cublasHandle_t cublasH);
	template
		shared_ptr<USImage> performRxBeamforming<float, int16_t>(
			shared_ptr<const USRawData> rawData,
			uint32_t subArraySize,
			uint32_t temporalSmoothing,
			cublasHandle_t cublasH);
	template
		shared_ptr<USImage> performRxBeamforming<float, float>(
			shared_ptr<const USRawData> rawData,
			uint32_t subArraySize,
			uint32_t temporalSmoothing,
			cublasHandle_t cublasH);
}
