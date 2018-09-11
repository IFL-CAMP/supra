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

using namespace std;

#ifdef HAVE_CUDA
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif
#endif

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
			uint32_t scanlineIdxStart, uint32_t subArraySize, uint8_t* subArrayMasks,
			uint32_t* subArraySizes, uint32_t* subArrayOffsets)
		{
			int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
			int scanlineIdxLocal = blockIdx.y;
			int scanlineIdx = scanlineIdxLocal + scanlineIdxStart;

			if (sampleIdx < numSamples)
			{
				int numSubArrays = numChannels - subArraySize + 1;

				uint32_t apertureFirst = findFirstValidElement(rawData, sampleIdx, scanlineIdx, numSamples, numChannels);
				uint32_t apertureLast = findLastValidElement(rawData, sampleIdx, scanlineIdx, numSamples, numChannels);

				subArraySizes[scanlineIdxLocal * numSamples + sampleIdx] = min(subArraySize, (apertureLast - apertureFirst + 1));

				for (uint32_t subArrayIdx = 0; subArrayIdx < numSubArrays; subArrayIdx++)
				{
					subArrayMasks[subArrayIdx + (scanlineIdxLocal * numSamples + sampleIdx)*numSubArrays] =
						(apertureFirst + subArrayIdx + subArraySizes[scanlineIdxLocal * numSamples + sampleIdx] - 1) <= apertureLast;
					subArrayOffsets[subArrayIdx + (scanlineIdxLocal * numSamples + sampleIdx)*numSubArrays] =
						min(max(apertureFirst + subArrayIdx, 0), numChannels - subArraySizes[scanlineIdxLocal * numSamples + sampleIdx] + 1);
					if (subArrayIdx > 1)
					{
						subArrayMasks[subArrayIdx + (scanlineIdxLocal * numSamples + sampleIdx)*numSubArrays] =
							subArrayMasks[subArrayIdx + (scanlineIdxLocal * numSamples + sampleIdx)*numSubArrays] &&
							(subArrayOffsets[subArrayIdx + (scanlineIdxLocal * numSamples + sampleIdx)*numSubArrays] !=
								subArrayOffsets[(subArrayIdx - 1) + (scanlineIdxLocal * numSamples + sampleIdx)*numSubArrays]);
					}
				}
			}
		}

		template <typename ChannelDataType>
		__global__ void computeRmatrices(const ChannelDataType* rawData,
			uint32_t numSamples, uint32_t numChannels, uint32_t scanlineIdxStart,
			uint32_t subArraySize, const uint8_t * subArrayMasks,
			const uint32_t* subArraySizes, const uint32_t* subArrayOffsets, float* Rmatrices)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			int sampleIdx = blockIdx.x;
			int scanlineIdxLocal = blockIdx.y;
			int scanlineIdx = scanlineIdxLocal + scanlineIdxStart;

			int numSubArrays = numChannels - subArraySize + 1;

			if (sampleIdx < numSamples)
			{
				int subArraySizeLocal = subArraySizes[scanlineIdxLocal * numSamples + sampleIdx];
				if (subArraySizeLocal > 0)
				{
					int numelR = subArraySizeLocal*subArraySizeLocal;
					float* R = &Rmatrices[(scanlineIdxLocal*numSamples + sampleIdx) * (subArraySize*(subArraySize+1)/2)];

					for (uint32_t subArrayIdx = 0; subArrayIdx < numSubArrays; subArrayIdx++)
					{
						if (subArrayMasks[subArrayIdx + (scanlineIdxLocal*numSamples + sampleIdx)*numSubArrays])
						{
							auto offset = subArrayOffsets[subArrayIdx + (scanlineIdxLocal*numSamples + sampleIdx)*numSubArrays];

							for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
							{
								int rowIdx = matrixIdx % subArraySizeLocal;
								int colIdx = matrixIdx / subArraySizeLocal;
								
								if (rowIdx <= colIdx) // A is symmetric!
								{
									int packedMatrixStorageIdx = rowIdx + colIdx*(colIdx + 1)/2;

									float xCol = readRawData(rawData, sampleIdx, offset + colIdx, scanlineIdx, numSamples, numChannels);
									float xRow = readRawData(rawData, sampleIdx, offset + rowIdx, scanlineIdx, numSamples, numChannels);
									atomicAdd(&R[packedMatrixStorageIdx], xCol*xRow);
								}
							}
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
			int sampleIdx = blockIdx.x;
			int scanlineIdxLocal = blockIdx.y;
			
			if (sampleIdx < numSamples)
			{
				int subArraySizeLocal = subArraySizes[scanlineIdxLocal * numSamples + sampleIdx];
				if (subArraySizeLocal > 0)
				{
					int numelR = subArraySizeLocal*(subArraySizeLocal + 1) /2;
					int numelRfull = subArraySize*(subArraySize + 1) /2;

					int firstIdx = max(0, sampleIdx - (int)(temporalSmoothing)) + scanlineIdxLocal * numSamples;
					int lastIdx = min((int)(numSamples)-1, sampleIdx + (int)(temporalSmoothing)) + scanlineIdxLocal * numSamples;

					float scaling = 1.0f;
					for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
					{
						float finalEntry = 0.0f;

						for (int tempIdx = firstIdx; tempIdx <= lastIdx; tempIdx++)
						{
							finalEntry += Rmatrices[matrixIdx + tempIdx*numelRfull];
						}
						TempRmatrices[matrixIdx + (scanlineIdxLocal * numSamples + sampleIdx)*numelRfull] = finalEntry*scaling;
					}
				}
			}
		}

		template <typename T>
		__inline__ __device__ T warpAllReduceSum(T val) {
			for (int mask = warpSize / 2; mask > 0; mask /= 2)
			{
				val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
			}
			return val;
		}

		template <typename T>
		__device__ void blockReduceWarpAtomic(T in, T* out) {
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
			int sampleIdx = blockIdx.x;
			int scanlineIdxLocal = blockIdx.y;
			
			if (sampleIdx < numSamples)
			{
				int subArraySizeLocal = subArraySizes[scanlineIdxLocal * numSamples + sampleIdx];
				if (subArraySizeLocal > 0)
				{
					int numelR = subArraySize*(subArraySize + 1) /2;

					float* R = &Rmatrices[(scanlineIdxLocal * numSamples + sampleIdx)*numelR];

					// compute trace in one block
					float localSum = 0.0f;
					for (int diagIdx = tIdx; diagIdx < subArraySizeLocal; diagIdx += blockDim.x*blockDim.y)
					{
						int packedMatrixStorageIdx = diagIdx + diagIdx*(diagIdx + 1)/2;

						localSum += R[packedMatrixStorageIdx];
					}
					float trace = warpAllReduceSum(localSum);
					float loading = (1.0f / static_cast<float>(subArraySizeLocal)) * trace;

					for (int diagIdx = tIdx; diagIdx < subArraySizeLocal; diagIdx += blockDim.x*blockDim.y)
					{
						int packedMatrixStorageIdx = diagIdx + diagIdx*(diagIdx + 1)/2;

						R[packedMatrixStorageIdx] += loading;
					}
				}
			}
		}

		template <typename T>
		__device__ void setVectorBlockWise(T* vect, const T& value, const uint32_t& numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				vect[elementIdx] = value;
			}
		}

		template <typename Tdst, typename Tsrc>
		__device__ void assignVectorBlockWise(Tdst* dst, const Tsrc* src, const uint32_t& numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				dst[elementIdx] = static_cast<Tdst>(src[elementIdx]);
			}
		}
		
		template <typename T>
		__device__ void assignMatrixBlockWise(T* dst, const T* src, const uint32_t& stride, const uint32_t& numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			uint32_t numElementsMatrix = numElements*numElements;
			
			for (int matrixIdx = tIdx; matrixIdx < numElementsMatrix; matrixIdx += blockDim.x*blockDim.y)
			{
				// We iterate over the rows first, to have less atomic writes to the same output element
				int rowIdx = matrixIdx % numElements;
				int colIdx = matrixIdx / numElements;

				if (rowIdx <= colIdx) // A is symmetric!
				{
					int matrixStorageIdx = colIdx + rowIdx * stride;
					dst[matrixStorageIdx] = src[matrixStorageIdx];
				}
			}
		}

		// Following LAPACK  UPLO = U storage
		template <typename T>
		__device__ void assignSymmetricMatrixBlockWise(T* dst, const T* src, const uint32_t& stride, const uint32_t& numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			uint32_t numElementsMatrix = numElements*(numElements + 1) / 2;
			
			for (int matrixIdx = tIdx; matrixIdx < numElementsMatrix; matrixIdx += blockDim.x*blockDim.y)
			{
				dst[matrixIdx] = src[matrixIdx];
			}
		}
		
		template <typename Tx, typename Ty>
		__device__ void saxpyVectorBlockWise(Tx* c, const Tx& a, const Tx* x, const Ty* y, const uint32_t& numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				c[elementIdx] = a*x[elementIdx] + y[elementIdx];
			}
		}
	
		template <typename T, typename matrixType>
		__device__ void applySymmetricMatrixBlockWise(T* vect, const matrixType* A, const T* x,
			const uint32_t& stride, const uint32_t& numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			uint32_t numElementsMatrix = numElements*numElements;

			setVectorBlockWise(vect, static_cast<T>(0), numElements);

			for (int matrixIdx = tIdx; matrixIdx < numElementsMatrix; matrixIdx += blockDim.x*blockDim.y)
			{
				// We iterate over the rows first, to have less atomic writes to the same output element
				int rowIdx = matrixIdx % numElements;
				int colIdx = matrixIdx / numElements;
				
				int packedMatrixStorageIdx = rowIdx + colIdx*(colIdx + 1)/2;
				if (rowIdx > colIdx) // A is symmetric!
				{
					packedMatrixStorageIdx = colIdx + rowIdx*(rowIdx + 1)/2;
				}

				//Addition needs to be atomic, because another thread can compute
				//the partial value of this row (from another column)
				atomicAdd(&vect[rowIdx], A[packedMatrixStorageIdx] * x[colIdx]);
			}
		}
		
		template <typename T, typename matrixType>
		__device__ void applySymmetricJacobiPreconditionerBlockWise(T* vect, const matrixType* M, const T* x,
			const uint32_t& stride, const uint32_t& numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				int packedMatrixStorageIdx = elementIdx + elementIdx*(elementIdx + 1)/2;

				vect[elementIdx] = x[elementIdx] / M[packedMatrixStorageIdx];
			}
		}

		template <typename T>
		__device__ T scalarProductBlockWise(const T* x, const T* y, T* scratch, const uint32_t& numElements)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			if (tIdx == 0)
			{
				*scratch = 0;
			}

			T sum = 0;
			for (int elementIdx = tIdx; elementIdx < numElements; elementIdx += blockDim.x*blockDim.y)
			{
				sum += x[elementIdx] * y[elementIdx];
			}
			blockReduceWarpAtomic(sum, scratch);

			return *scratch;
		}

		template <typename pcgWorkType, typename inputType, typename outputType>
		__global__ void solveBlockwisePCGJacobi(
			const inputType* Amatrices, const uint32_t* systemSizes, const inputType* rightHandSides, 
			uint32_t stride, uint32_t numSystems, int maxIterations, pcgWorkType convergenceThreshold, outputType* solutions)
		{
			int systemIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

			if (systemIdx < numSystems)
			{
				uint32_t systemSize = systemSizes[systemIdx];

				// We use the following vectors: x, b, r, d, q, s
				// From them, we hold in shared memory: x, r, d, q, s
				// That means we need 4* systemSize + 1 of pcgWorkType and (systemSize * (systemSize + 1) /2) of inputType in shared memory
				extern __shared__ pcgWorkType smem[];
				pcgWorkType* shX = smem;
				pcgWorkType* shR = smem + systemSize;
				pcgWorkType* shD = smem + 2 * systemSize;
				pcgWorkType* shQ = smem + 3 * systemSize;
				pcgWorkType* shScratch = smem + 4 * systemSize;
				inputType* shA = reinterpret_cast<inputType*>(shScratch + 1); 
				
				// s and q are independent and never needed at the same time -> they can share storage
				pcgWorkType* shS = shQ;

				const inputType* b = &rightHandSides[systemIdx * stride];
				const inputType* A = &Amatrices[systemIdx * (stride*(stride+1)/2)];
				
				assignSymmetricMatrixBlockWise(shA, A, stride, systemSize);
				
				outputType* solution = &solutions[systemIdx * stride];

				pcgWorkType deltaNew;
				pcgWorkType delta0;
				pcgWorkType alpha;
				pcgWorkType beta;
				
				// Initialization
				// x = ones
				setVectorBlockWise(shX, static_cast<pcgWorkType>(1.0), systemSize);
				// r = b - Ax
				{
					// because d will be set in the next assignment, we can use it as tmp
					auto shTmp = shD;
					// tmp = Ax
					applySymmetricMatrixBlockWise(shTmp, shA, shX, stride, systemSize);

					// r = b - tmp
					saxpyVectorBlockWise(shR, static_cast<pcgWorkType>(-1.0), shTmp, b, systemSize);
				}
				// d = M^-1 r
				applySymmetricJacobiPreconditionerBlockWise(shD, shA, shR, stride, systemSize);
				// deltaNew = r*d;
				deltaNew = scalarProductBlockWise(shR, shD, shScratch, systemSize);
				delta0 = deltaNew;

				for (int i = 0; i < maxIterations && deltaNew > convergenceThreshold*delta0; i++)
				{
					//q = Ad;
					applySymmetricMatrixBlockWise(shQ, shA, shD, stride, systemSize);
					//alpha = deltaNew / (d * q);
					alpha = deltaNew / scalarProductBlockWise(shD, shQ, shScratch, systemSize);
					//x = x + alpha d;
					saxpyVectorBlockWise(shX, alpha, shD, shX, systemSize);

					//r = r - alpha q;
					saxpyVectorBlockWise(shR, -alpha, shQ, shR, systemSize);

					//s = M^-1 r;
					applySymmetricJacobiPreconditionerBlockWise(shS, shA, shR, stride, systemSize);
					
					pcgWorkType deltaOld = deltaNew;

					//deltaNew = r * s;
					deltaNew = scalarProductBlockWise(shR, shS, shScratch, systemSize);

					beta = deltaNew / deltaOld;
					
					//d = s + beta d;
					saxpyVectorBlockWise(shD, beta, shD, shS, systemSize);
				}

				// solution = x;
				assignVectorBlockWise(solution, shX, systemSize);
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
			uint32_t scanlineIdxStart,
			uint32_t subArraySize,
			const uint8_t * subArrayMasks,
			const uint32_t * subArraySizes,
			const uint32_t * subArrayOffsets,
			float outputClamp,
			ImageDataType* beamformed)
		{
			int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
			int sampleIdx = blockIdx.x;
			int scanlineIdxLocal = blockIdx.y;
			int scanlineIdx = scanlineIdxLocal + scanlineIdxStart;

			if (sampleIdx < numSamples)
			{
				int numSubArrays = numChannels - subArraySize + 1;
				int subArraySizeLocal = subArraySizes[scanlineIdxLocal*numSamples + sampleIdx];

				int numSubArraysActive = 0;
				for (int subArrayIdx = 0; subArrayIdx < numSubArrays; subArrayIdx++)
				{
					if (subArrayMasks[subArrayIdx + (scanlineIdxLocal*numSamples + sampleIdx)*numSubArrays] != 0)
					{
						numSubArraysActive++;
					}
				}

				// compute weight scaling <a, R\a>
				const float* RinvAloc = &RinverseA[(scanlineIdxLocal*numSamples + sampleIdx) * subArraySize];
				const float* Aloc = &A[(scanlineIdxLocal*numSamples + sampleIdx) * subArraySize];
				float weightScaling = 0.0f;
				for (int vectorIdx = tIdx; vectorIdx < subArraySizeLocal; vectorIdx += blockDim.x*blockDim.y)
				{
					weightScaling += RinvAloc[vectorIdx] * Aloc[vectorIdx];
				}
				weightScaling = 1.0f / (warpAllReduceSum(weightScaling) * powf(numSubArraysActive, 1.2f));

				// compute one sample at a time, according to spatial smoothing
				float beamformedSample = 0.0f;
				for (int vectorIdx = tIdx; vectorIdx < subArraySizeLocal; vectorIdx += blockDim.x*blockDim.y)
				{
					float sample = 0.0;
					for (int subArrayIdx = 0; subArrayIdx < numSubArrays; subArrayIdx++)
					{
						if (subArrayMasks[subArrayIdx + (scanlineIdxLocal*numSamples + sampleIdx)*numSubArrays] != 0)
						{
							auto offset = subArrayOffsets[subArrayIdx + (scanlineIdxLocal*numSamples + sampleIdx)*numSubArrays];
							sample += readRawData(rawData, sampleIdx, offset + vectorIdx, scanlineIdx, numSamples, numChannels);
						}
					}
					beamformedSample += sample * RinvAloc[vectorIdx] * weightScaling;
				}
				beamformedSample = warpAllReduceSum(beamformedSample);
				if (tIdx == 0)
				{
					if (abs(beamformedSample) > 1e7 || !::isfinite(beamformedSample))
					{
						beamformedSample = 0.0f;
					}
					float outputValue = min(max(beamformedSample, -outputClamp), outputClamp);
					beamformed[scanlineIdx + sampleIdx * numScanlines] =
						clampCast<ImageDataType>(outputValue);
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
			double convergenceThreshold,
			double outputClamp)
		{
			uint32_t scanlineBlockSize = 32;

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

			size_t numelRmatrices = (subArraySize*(subArraySize+1) /2)* numSamples * scanlineBlockSize;
			size_t numelVectors = subArraySize*numSamples* scanlineBlockSize;
			size_t numelSamples = numSamples* scanlineBlockSize;
			size_t numelSubArrayVectors = numSubArrays*numSamples* scanlineBlockSize;
			shared_ptr<Container<float> > Rmatrices =
				std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, numelRmatrices);
			shared_ptr<Container<float> > RmatricesTempSmooth =
				std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, numelRmatrices);
			shared_ptr<Container<float> > Avectors =
				std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, std::vector<float>(numelVectors, 1.0f));
			shared_ptr<Container<float> > Wvectors =
				std::make_shared<Container<float> >(ContainerLocation::LocationGpu, stream, numelVectors);
			shared_ptr<Container<uint8_t> > subArrayMasks =
				std::make_shared<Container<uint8_t> >(ContainerLocation::LocationGpu, stream, numelSubArrayVectors);
			shared_ptr<Container<uint32_t> > subArrayOffsets =
				std::make_shared<Container<uint32_t> >(ContainerLocation::LocationGpu, stream, numelSubArrayVectors);
			shared_ptr<Container<uint32_t> > subArraySizes = 
				std::make_shared<Container<uint32_t> >(ContainerLocation::LocationGpu, stream, numelSamples);
			
			// DEBUG
			//shared_ptr<Container<float> > allMatrices =
			//	std::make_shared<Container<float> >(ContainerLocation::LocationHost, stream, subArraySize*subArraySize * numSamples * numScanlines);

			for (uint32_t scanlineIdx = 0; scanlineIdx < numScanlines; scanlineIdx += scanlineBlockSize)
			{
				uint32_t numScanlinesBatch = min(scanlineBlockSize, numScanlines - scanlineIdx);

				cudaSafeCall(cudaMemsetAsync(Rmatrices->get(), 0, numelRmatrices * sizeof(float), stream));
				cudaSafeCall(cudaMemsetAsync(RmatricesTempSmooth->get(), 0, numelRmatrices * sizeof(float), stream));
				cudaSafeCall(cudaMemsetAsync(Wvectors->get(), 0, numelVectors * sizeof(float), stream));
				cudaSafeCall(cudaMemsetAsync(subArrayMasks->get(), 0, numelSubArrayVectors * sizeof(uint8_t), stream));
				cudaSafeCall(cudaMemsetAsync(subArrayOffsets->get(), 0, numelSubArrayVectors * sizeof(uint32_t), stream));
				cudaSafeCall(cudaMemsetAsync(subArraySizes->get(), 0, numelSamples * sizeof(uint32_t), stream));

				dim3 blockSize(32, 1);
				dim3 gridSize(numSamples, numScanlinesBatch);

				// determine how large the subarrays will be depending on the aperture and which are active
				dim3 gridSizeMasks((numSamples + blockSize.x - 1) / blockSize.x, numScanlinesBatch);
				computeSubArrayMasks <<<gridSizeMasks, blockSize, 0, stream >>> (
					gRawData->get(),
					numSamples,
					numChannels,
					scanlineIdx,
					subArraySize,
					subArrayMasks->get(),
					subArraySizes->get(),
					subArrayOffsets->get());
				cudaSafeCall(cudaPeekAtLastError());
				
				// Compute the covariance matrices
				computeRmatrices <<<gridSize, blockSize, 0, stream >>> (
					gRawData->get(),
					numSamples,
					numChannels,
					scanlineIdx,
					subArraySize,
					subArrayMasks->get(),
					subArraySizes->get(),
					subArrayOffsets->get(),
					Rmatrices->get()
					);
				cudaSafeCall(cudaPeekAtLastError());
				
				/*cudaSafeCall(cudaDeviceSynchronize());
				auto RmatricesHost =
					std::make_shared<Container<float> >(ContainerLocation::LocationHost, *Rmatrices);
				std::copy_n(RmatricesHost->get(), subArraySize*subArraySize*numSamplesBatch, allMatrices->get() + (scanlineIdx * numSamples + sampleIdx)*subArraySize*subArraySize);*/				
									
				// Smooth the covariance matrices
				computeTemporalSmoothRmatrices <<<gridSize, blockSize, 0, stream >>> (
					Rmatrices->get(),
					numSamples,
					subArraySize,
					numSubArrays,
					subArraySizes->get(),
					temporalSmoothing,
					RmatricesTempSmooth->get()
					);
				cudaSafeCall(cudaPeekAtLastError());
				
				/*cudaSafeCall(cudaDeviceSynchronize());
				auto RmatricesTempSmoothHost =
					std::make_shared<Container<float> >(ContainerLocation::LocationHost, *RmatricesTempSmooth);
				std::copy_n(RmatricesTempSmoothHost->get(), subArraySize*subArraySize*numSamplesBatch, allMatrices->get() + (scanlineIdx * numSamples + sampleIdx)*subArraySize*subArraySize);*/
									
				// Improve condition of matrices
				addDiagonalLoading <<<gridSize, dim3(32, 1), 0, stream >>> (
					RmatricesTempSmooth->get(),
					numSamples, subArraySize,
					subArraySizes->get()
					);
				cudaSafeCall(cudaPeekAtLastError());
				
				/*cudaSafeCall(cudaDeviceSynchronize());
				auto RmatricesTempSmoothHost =
					std::make_shared<Container<float> >(ContainerLocation::LocationHost, *RmatricesTempSmooth);
				std::copy_n(RmatricesTempSmoothHost->get(), subArraySize*subArraySize*numSamplesBatch, allMatrices->get() + (scanlineIdx * numSamples + sampleIdx)*subArraySize*subArraySize);*/
				
				// solve for the beamforming weights with PCG
				typedef double pcgWorkType;
				//typedef float pcgWorkType;
				size_t sharedMemorySize = (4 * subArraySize + 1) * sizeof(pcgWorkType) + (subArraySize * (subArraySize + 1) /2) * sizeof(float);
				solveBlockwisePCGJacobi<pcgWorkType, float, float> <<<gridSize, blockSize, sharedMemorySize, stream >>>(
					RmatricesTempSmooth->get(),
					subArraySizes->get(),
					Avectors->get(),
					subArraySize,
					numSamples * numScanlinesBatch,
					maxIterations,
					convergenceThreshold,
					Wvectors->get()
				);
				cudaSafeCall(cudaPeekAtLastError());
				
				// calculate beamforming weights from the solutions and perform beamforming
				applyWeights <<<gridSize, dim3(32, 1), 0, stream >>> (
					Wvectors->get(),
					Avectors->get(),
					gRawData->get(),
					numSamples,
					numChannels,
					numScanlines,
					scanlineIdx,
					subArraySize,
					subArrayMasks->get(),
					subArraySizes->get(),
					subArrayOffsets->get(),
					outputClamp,
					pData->get()
					);
				cudaSafeCall(cudaPeekAtLastError());
			}

			auto retImage = std::make_shared<USImage>(
				vec2s{ numScanlines, numSamples },
				pData,
				rawData->getImageProperties(),
				rawData->getReceiveTimestamp(),
				rawData->getSyncTimestamp());
			/*auto retImage = std::make_shared<USImage>(
				vec3s{ subArraySize*subArraySize, numSamples, numScanlines },
				allMatrices,
				rawData->getImageProperties(),
				rawData->getReceiveTimestamp(),
				rawData->getSyncTimestamp());*/

			return retImage;
		}

		template
			shared_ptr<USImage> performRxBeamforming<int16_t, int16_t>(
				shared_ptr<const USRawData> rawData,
				uint32_t subArraySize,
				uint32_t temporalSmoothing,
				uint32_t maxIterations,
				double convergenceThreshold,
				double outputClamp);
		template
			shared_ptr<USImage> performRxBeamforming<int16_t, float>(
				shared_ptr<const USRawData> rawData,
				uint32_t subArraySize,
				uint32_t temporalSmoothing,
				uint32_t maxIterations,
				double convergenceThreshold,
				double outputClamp);
		template
			shared_ptr<USImage> performRxBeamforming<float, int16_t>(
				shared_ptr<const USRawData> rawData,
				uint32_t subArraySize,
				uint32_t temporalSmoothing,
				uint32_t maxIterations,
				double convergenceThreshold,
				double outputClamp);
		template
			shared_ptr<USImage> performRxBeamforming<float, float>(
				shared_ptr<const USRawData> rawData,
				uint32_t subArraySize,
				uint32_t temporalSmoothing,
				uint32_t maxIterations,
				double convergenceThreshold,
				double outputClamp);
	}
}
