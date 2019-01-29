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

#include "RxBeamformerMVpcg.h"

#include "USImage.h"
#include "USRawData.h"

#include <utilities/Logging.h>
#include <utilities/cudaUtility.h>
#include <utilities/cublasUtility.h>

using namespace std;

namespace supra
{
	namespace RxBeamformerMVpcg
	{
		template <typename ChannelDataType>
		inline __device__ ChannelDataType 
			readRawData(
				const ChannelDataType* rawData,
				uint32_t sampleIdx, uint32_t channelIdx, uint32_t scanlineIdx,
				uint32_t numSamples, uint32_t numChannels)
		{
			//TEST
		/*	if (sampleIdx >= numSamples || channelIdx >= numChannels)
			{
				printf("readRawData OOB: %d, %d, %d, %d, %d\n", sampleIdx, channelIdx, scanlineIdx,
					numSamples, numChannels);
			}*/
			return rawData[sampleIdx + channelIdx*numSamples + scanlineIdx*numChannels*numSamples];
		}

		template <typename ChannelDataType>
		__device__ uint32_t findFirstValidElement(
			const ChannelDataType* rawData,
			uint32_t sampleIdx, uint32_t scanlineIdx,
			uint32_t numSamples, uint32_t numChannels)
		{
			uint32_t firstValid = 0;
			for (uint32_t channelIdx = 0; channelIdx < numChannels; channelIdx++)
			{
				if (readRawData(rawData, sampleIdx, channelIdx, scanlineIdx, numSamples, numChannels) != 0)
				{
					firstValid = channelIdx;
					break;
				}
			}
			return firstValid;
		}

		template <typename ChannelDataType>
		__device__ uint32_t findLastValidElement(
			const ChannelDataType* rawData,
			uint32_t sampleIdx, uint32_t scanlineIdx,
			uint32_t numSamples, uint32_t numChannels)
		{
			uint32_t lastValid = 0;
			for (int32_t channelIdx = numChannels - 1; channelIdx >= 0; channelIdx--)
			{
				if (readRawData(rawData, sampleIdx, channelIdx, scanlineIdx, numSamples, numChannels) != 0)
				{
					lastValid = channelIdx;
					break;
				}
			}
			return lastValid;
		}

		template <typename ChannelDataType>
		__global__ void computeSubArrayMasks(
			const ChannelDataType* rawData,
			uint32_t numSamples, uint32_t numChannels,
			uint32_t scanlineIdx, uint32_t sampleIdxStart,
			uint32_t subArraySize, uint8_t* subArrayMasks,
			uint32_t* subArraySizes, uint32_t* subArrayOffsets)
		{
			int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;

			if (sampleIdx < numSamples)
			{
				int numSubArrays = numChannels - subArraySize + 1;

				uint32_t apertureFirst = findFirstValidElement(rawData, sampleIdx, scanlineIdx, numSamples, numChannels);
				uint32_t apertureLast = findLastValidElement(rawData, sampleIdx, scanlineIdx, numSamples, numChannels);

				subArraySizes[sampleIdx] = min(subArraySize, (apertureLast - apertureFirst + 1));

				for (uint32_t subArrayIdx = 0; subArrayIdx < numSubArrays; subArrayIdx++)
				{
					subArrayMasks[subArrayIdx + sampleIdx*numSubArrays] =
						(apertureFirst + subArrayIdx + subArraySizes[sampleIdx] - 1) <= apertureLast;
					subArrayOffsets[subArrayIdx + sampleIdx*numSubArrays] =
						min(max(apertureFirst + subArrayIdx, 0), numChannels - subArraySizes[sampleIdx] + 1);
					if (subArrayIdx > 1)
					{
						subArrayMasks[subArrayIdx + sampleIdx*numSubArrays] =
							subArrayMasks[subArrayIdx + sampleIdx*numSubArrays] &&
							(subArrayOffsets[subArrayIdx + sampleIdx*numSubArrays] !=
								subArrayOffsets[(subArrayIdx - 1) + sampleIdx*numSubArrays]);
					}
				}
			}
		}

		template <typename ChannelDataType>
		__global__ void computeRmatrices(const ChannelDataType* rawData,
			uint32_t numSamples, uint32_t numChannels, uint32_t scanlineIdx, uint32_t sampleIdxStart,
			uint32_t subArraySize, const uint8_t * subArrayMasks,
			const uint32_t* subArraySizes, const uint32_t* subArrayOffsets, float* Rmatrices)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			int sampleIdxLocal = (blockIdx.y * gridDim.x) + blockIdx.x;
			int sampleIdx = sampleIdxLocal + sampleIdxStart;

			int numSubArrays = numChannels - subArraySize + 1;

			if (sampleIdx < numSamples)
			{
				int subArraySizeLocal = subArraySizes[sampleIdx];
				int numelR = subArraySizeLocal*subArraySizeLocal;
				float* R = &Rmatrices[sampleIdxLocal * subArraySize*subArraySize];

				for (uint32_t subArrayIdx = 0; subArrayIdx < numSubArrays; subArrayIdx++)
				{
					if (subArrayMasks[subArrayIdx + sampleIdx*numSubArrays])
					{
						auto offset = subArrayOffsets[subArrayIdx + sampleIdx*numSubArrays];

						for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
						{
							int colIdx = matrixIdx % subArraySizeLocal;
							int rowIdx = matrixIdx / subArraySizeLocal;

							float xCol = readRawData(rawData, sampleIdx, offset + colIdx, scanlineIdx, numSamples, numChannels);
							float xRow = readRawData(rawData, sampleIdx, offset + rowIdx, scanlineIdx, numSamples, numChannels);

							int matrixStorageIdx = colIdx + rowIdx * subArraySize;

							atomicAdd(&R[matrixStorageIdx], xCol*xRow);
						}
					}
				}
			}
		}

		__global__ void computeTemporalSmoothRmatrices(const float* Rmatrices,
			uint32_t numSamples, uint32_t subArraySize, uint32_t numSubArrays,
			const uint32_t* subArraySizes, uint32_t temporalSmoothing, float* TempRmatrices)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			int sampleIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

			if (sampleIdx < numSamples)
			{
				int subArraySizeLocal = subArraySizes[sampleIdx];
				int numelR = subArraySizeLocal*subArraySizeLocal;
				int numelRfull = subArraySize*subArraySize;

				int firstIdx = max(0, sampleIdx - (int)(temporalSmoothing));
				int lastIdx = min((int)(numSamples)-1, sampleIdx + (int)(temporalSmoothing));

				float scaling = 1.0f;
				for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
				{
					float finalEntry = 0.0f;
					for (int tempIdx = firstIdx; tempIdx <= lastIdx; tempIdx++)
					{
						finalEntry += Rmatrices[matrixIdx + tempIdx*numelRfull];
					}

					int colIdx = matrixIdx % subArraySizeLocal;
					int rowIdx = matrixIdx / subArraySizeLocal;
					int matrixStorageIdx = colIdx + rowIdx * subArraySize;
					TempRmatrices[matrixStorageIdx + sampleIdx*numelRfull] = finalEntry*scaling;
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

		template <typename T>
		__device__ void blockReduceWarpAtomic(T in, T* out) {
			// We only add to the output, so set it to zero first
			if (threadIdx.x == 0)
			{
				*out = 0;
			}
			auto warpSum = warpAllReduceSum(in);
			if ((threadIdx.x & (warpSize - 1)) == 0)
			{
				atomicAdd(out, warpSum);
			}
			__syncthreads();
		}

		__global__ void addDiagonalLoading(float* Rmatrices,
			uint32_t numSamples, uint32_t subArraySize, const uint32_t* subArraySizes)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			int sampleIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

			if (sampleIdx < numSamples)
			{
				int subArraySizeLocal = subArraySizes[sampleIdx];
				int numelRfull = subArraySize*subArraySize;

				float* R = &Rmatrices[sampleIdx*numelRfull];

				// compute trace in one block
				float localSum = 0.0f;
				for (int diagIdx = tIdx; diagIdx < subArraySizeLocal; diagIdx += blockDim.x*blockDim.y)
				{
					// subArraySize + 1 (instead of subArraySize) to follow the diagonal
					int matrixIdx = diagIdx * (subArraySize + 1);

					localSum += R[matrixIdx];
				}
				float trace = warpAllReduceSum(localSum);
				float loading = (1.0f / static_cast<float>(subArraySizeLocal)) * trace;

				for (int diagIdx = tIdx; diagIdx < subArraySizeLocal; diagIdx += blockDim.x*blockDim.y)
				{
					// subArraySize + 1 (instead of subArraySize) to follow the diagonal
					int matrixIdx = diagIdx * (subArraySize + 1);

					R[matrixIdx] += loading;
				}
			}
		}

		__device__ void setVectorBlockWise(float* vect, float value, uint32_t numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				vect[elementIdx] = value;
			}
		}

		__device__ void assignVectorBlockWise(const float* src, float* dst, uint32_t numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				dst[elementIdx] = src[elementIdx];
			}
		}

		__device__ void saxpyVectorBlockWise(float* c, float a, const float* x, const float* y, uint32_t numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				c[elementIdx] = a*x[elementIdx] + y[elementIdx];
			}
		}

		__device__ void applyMatrixBlockWise(float* vect, const float* A, const float* x,
			uint32_t stride, uint32_t numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			// Because we only add atomically to the output, we need to set the target vector to 0 first
			setVectorBlockWise(vect, 0.0f, numElements);

			uint32_t numElementsMatrix = numElements*numElements;
			for (int matrixIdx = tIdx; matrixIdx < numElementsMatrix; matrixIdx += blockDim.x*blockDim.y)
			{
				// We iterate over the rows first, to have less atomic writes to the same output element
				int rowIdx = matrixIdx % numElements;
				int colIdx = matrixIdx / numElements;

				int matrixStorageIdx = colIdx + rowIdx * stride;

				//Addition needs to be atomic, because another thread can compute
				//the partial value of this row (from another column)
				atomicAdd(&vect[rowIdx], A[matrixStorageIdx] * x[colIdx]);
			}
		}

		__device__ void applyJacobiPreconditionerBlockWise(float* vect, const float* M, const float* x,
			uint32_t stride, uint32_t numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				// stride + 1, because we want to access only the diagonal
				int matrixStorageIdx = elementIdx * (stride + 1);

				vect[elementIdx] = x[elementIdx] / M[matrixStorageIdx];
			}
		}

		__device__ float scalarProductBlockWise(const float* x, const float* y, float* scratch, uint32_t numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;

			float sum = 0.0f;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				sum += x[elementIdx] * y[elementIdx];
			}
			blockReduceWarpAtomic(sum, scratch);

			return *scratch;
		}

		__global__ void solveBlockwisePCGJacobi(
			const float* Amatrices, const uint32_t* systemSizes, const float* rightHandSides, 
			uint32_t stride, uint32_t numSystems, int maxIterations, float convergenceThreshold, float* solutions)
		{
			int systemIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

			if (systemIdx < numSystems)
			{
				uint32_t systemSize = systemSizes[systemIdx];

				// We use the following vectors: x, b, r, d, q, s
				// From them, we hold in shared memory: x, r, d, q, s
				// That means we need 5* systemSize + 1 floats in shared memory
				extern __shared__ float smem[];
				float* shX = smem;
				float* shR = smem + systemSize;
				float* shD = smem + 2 * systemSize;
				float* shQ = smem + 3 * systemSize;
				float* shS = smem + 4 * systemSize;
				float* shScratch = smem + 5 * systemSize;

				const float* b = &rightHandSides[systemIdx * stride];
				const float* A = &Amatrices[systemIdx * stride*stride];

				float* solution = &solutions[systemIdx * stride];

				float deltaNew;
				float delta0;
				float alpha;
				float beta;
				
				// Initialization
				// x = ones
				setVectorBlockWise(shX, 1.0f, systemSize);
				// r = b - Ax
				{
					// because d will be set in the next assignment, we can use it as tmp
					auto shTmp = shD;
					// tmp = Ax
					applyMatrixBlockWise(shTmp, A, shX, stride, systemSize);

					// r = b - tmp
					saxpyVectorBlockWise(shR, -1.0f, shTmp, b, systemSize);
				}
				// d = M^-1 r
				applyJacobiPreconditionerBlockWise(shD, A, shR, stride, systemSize);
				// deltaNew = r*d;
				deltaNew = scalarProductBlockWise(shR, shD, shScratch, systemSize);
				delta0 = deltaNew;
				for (int i = 0; i < maxIterations && deltaNew > convergenceThreshold*delta0; i++)
				{
					//q = Ad;
					applyMatrixBlockWise(shQ, A, shD, stride, systemSize);
					//alpha = deltaNew / (d * q);
					alpha = deltaNew / scalarProductBlockWise(shD, shQ, shScratch, systemSize);
					//x = x + alpha d;
					saxpyVectorBlockWise(shX, alpha, shD, shX, systemSize);

					//r = r - alpha q;
					saxpyVectorBlockWise(shR, -alpha, shQ, shR, systemSize);

					//s = M^-1 r;
					applyJacobiPreconditionerBlockWise(shS, A, shR, stride, systemSize);
					
					float deltaOld = deltaNew;

					//deltaNew = r * s;
					deltaNew = scalarProductBlockWise(shR, shS, shScratch, systemSize);

					beta = deltaNew / deltaOld;
					
					//d = s + beta d;
					saxpyVectorBlockWise(shD, beta, shD, shS, systemSize);
				}

				// solution = x;
				assignVectorBlockWise(shX, solution, systemSize);
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
			const uint32_t * subArraySizes,
			const uint32_t * subArrayOffsets,
			ImageDataType* beamformed)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			int sampleIdxLocal = (blockIdx.y * gridDim.x) + blockIdx.x;
			int sampleIdx = sampleIdxLocal + sampleIdxStart;

			if (sampleIdx < numSamples)
			{
				int numSubArrays = numChannels - subArraySize + 1;
				int subArraySizeLocal = subArraySizes[sampleIdx];

				int numSubArraysActive = 0;
				for (int subArrayIdx = 0; subArrayIdx < numSubArrays; subArrayIdx++)
				{
					if (subArrayMasks[subArrayIdx + sampleIdxLocal*numSubArrays] != 0)
					{
						numSubArraysActive++;
					}
				}

				// compute weight scaling <a, R\a>
				const float* RinvAloc = &RinverseA[sampleIdxLocal * subArraySize];
				const float* Aloc = &A[sampleIdxLocal * subArraySize];
				float weightScaling = 0.0f;
				for (int vectorIdx = tIdx; vectorIdx < subArraySizeLocal; vectorIdx += blockDim.x*blockDim.y)
				{
					weightScaling += RinvAloc[vectorIdx] * Aloc[vectorIdx];
				}
				weightScaling = 1.0f / (warpAllReduceSum(weightScaling) * numSubArraysActive);

				// compute one sample at a time, according to spatial smoothing
				float beamformedSample = 0.0f;
				for (int vectorIdx = tIdx; vectorIdx < subArraySizeLocal; vectorIdx += blockDim.x*blockDim.y)
				{
					float sample = 0.0;
					for (int subArrayIdx = 0; subArrayIdx < numSubArrays; subArrayIdx++)
					{
						if (subArrayMasks[subArrayIdx + sampleIdxLocal*numSubArrays] != 0)
						{
							auto offset = subArrayOffsets[subArrayIdx + sampleIdx*numSubArrays];
							sample += readRawData(rawData, sampleIdx, offset + vectorIdx, scanlineIdx, numSamples, numChannels);
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
			uint32_t maxIterations,
			double convergenceThreshold)
		{
			uint32_t sampleBlockSize = 2000;//128;

			//Ensure the raw-data are on the gpu
			auto gRawData = rawData->getData<ChannelDataType>();
			if (!gRawData->isGPU() && !gRawData->isBoth())
			{
				gRawData = std::make_shared<Container<ChannelDataType> >(LocationGpu, *gRawData);
			}
			auto stream = gRawData->getStream();

			uint32_t numScanlines = static_cast<uint32_t>(rawData->getNumScanlines());
			uint32_t numSamples = static_cast<uint32_t>(rawData->getNumSamples());
			uint32_t numChannels = static_cast<uint32_t>(rawData->getNumReceivedChannels());
			if (subArraySize == 0)
			{
				subArraySize = numChannels / 2;
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
			shared_ptr<Container<float> > Wvectors =
				std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, std::vector<float>(subArraySize*sampleBlockSize, 5.0f));
			shared_ptr<Container<uint8_t> > subArrayMasks =
				std::make_shared<Container<uint8_t> >(ContainerLocation::LocationGpu, stream, numSubArrays*sampleBlockSize);
			shared_ptr<Container<uint32_t> > subArraySizes = 
				std::make_shared<Container<uint32_t> >(ContainerLocation::LocationGpu, stream, sampleBlockSize);
			shared_ptr<Container<uint32_t> > subArrayOffsets =
				std::make_shared<Container<uint32_t> >(ContainerLocation::LocationGpu, stream, numSubArrays*sampleBlockSize);

			for (uint32_t scanlineIdx = 0; scanlineIdx < numScanlines; scanlineIdx++)
			{
				for (uint32_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx += sampleBlockSize)
				{
					uint32_t numSamplesBatch = min(sampleBlockSize, numSamples - sampleIdx);

					cudaSafeCall(cudaMemsetAsync(Rmatrices->get(), 0, numelRmatrices * sizeof(float), stream));
					cudaSafeCall(cudaMemsetAsync(Wvectors->get(), 0, subArraySize*sampleBlockSize * sizeof(float), stream));
					cudaSafeCall(cudaMemsetAsync(subArrayMasks->get(), 0, numSubArrays*sampleBlockSize * sizeof(uint8_t), stream));
					cudaSafeCall(cudaMemsetAsync(subArraySizes->get(), 0, sampleBlockSize * sizeof(uint32_t), stream));
					cudaSafeCall(cudaMemsetAsync(subArrayOffsets->get(), 0, numSubArrays*sampleBlockSize * sizeof(uint32_t), stream));

					dim3 blockSize(32, 1);
					dim3 gridSize(numSamplesBatch, 1);

					// determine how large the subarrays will be depending on the aperture and which are active
					dim3 gridSizeMasks((numSamplesBatch + blockSize.x - 1) / blockSize.x, 1);
					computeSubArrayMasks << <gridSizeMasks, blockSize, 0, stream >> > (
						gRawData->get(),
						numSamples,
						numChannels,
						scanlineIdx,
						sampleIdx,
						subArraySize,
						subArrayMasks->get(),
						subArraySizes->get(),
						subArrayOffsets->get());
					cudaSafeCall(cudaPeekAtLastError());
					
					// Compute the covariance matrices
					computeRmatrices << <gridSize, blockSize, 0, stream >> > (
						gRawData->get(),
						numSamples,
						numChannels,
						scanlineIdx,
						sampleIdx,
						subArraySize,
						subArrayMasks->get(),
						subArraySizes->get(),
						subArrayOffsets->get(),
						Rmatrices->get()
						);
					cudaSafeCall(cudaPeekAtLastError());
					
					// Smooth the covariance matrices
					computeTemporalSmoothRmatrices << <gridSize, blockSize, 0, stream >> > (
						Rmatrices->get(),
						numSamplesBatch,
						subArraySize,
						numSubArrays,
						subArraySizes->get(),
						temporalSmoothing,
						RmatricesTempSmooth->get()
						);
					cudaSafeCall(cudaPeekAtLastError());
					
					// Improve condition of matrices
					addDiagonalLoading << <gridSize, dim3(32, 1), 0, stream >> > (
						RmatricesTempSmooth->get(),
						numSamplesBatch, subArraySize,
						subArraySizes->get()
						);
					cudaSafeCall(cudaPeekAtLastError());
					
					// solve for the beamforming weights with PCG
					size_t sharedMemorySize = (5 * subArraySize + 1) * sizeof(float);
					solveBlockwisePCGJacobi << <gridSize, blockSize, sharedMemorySize, stream >> >(
						RmatricesTempSmooth->get(),
						subArraySizes->get(),
						Avectors->get(),
						subArraySize,
						numSamplesBatch,
						maxIterations,
						(float)convergenceThreshold,
						Wvectors->get()
					);
					cudaSafeCall(cudaPeekAtLastError());
					
					// calculate beamforming weights from the solutions and perform beamforming
					applyWeights << <gridSize, dim3(32, 1), 0, stream >> > (
						Wvectors->get(),
						Avectors->get(),
						gRawData->get(),
						numSamples,
						numChannels,
						numScanlines,
						scanlineIdx,
						sampleIdx,
						subArraySize,
						subArrayMasks->get(),
						subArraySizes->get(),
						subArrayOffsets->get(),
						pData->get()
						);
					cudaSafeCall(cudaPeekAtLastError());
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
				uint32_t maxIterations,
				double convergenceThreshold);
		template
			shared_ptr<USImage> performRxBeamforming<int16_t, float>(
				shared_ptr<const USRawData> rawData,
				uint32_t subArraySize,
				uint32_t temporalSmoothing,
				uint32_t maxIterations,
				double convergenceThreshold);
		template
			shared_ptr<USImage> performRxBeamforming<float, int16_t>(
				shared_ptr<const USRawData> rawData,
				uint32_t subArraySize,
				uint32_t temporalSmoothing,
				uint32_t maxIterations,
				double convergenceThreshold);
		template
			shared_ptr<USImage> performRxBeamforming<float, float>(
				shared_ptr<const USRawData> rawData,
				uint32_t subArraySize,
				uint32_t temporalSmoothing,
				uint32_t maxIterations,
				double convergenceThreshold);
	}
}
