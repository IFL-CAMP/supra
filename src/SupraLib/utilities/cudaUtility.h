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

#ifndef __CUDAUTILITY_H__
#define __CUDAUTILITY_H__

#include <cmath>
#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#ifdef HAVE_CUFFT
#include <cufft.h>
#endif
#endif
#include <cstdio>
#include "utilities/Logging.h"
#include <algorithm>
#include <cmath>
#include <cfloat>

namespace supra
{
#ifdef __CUDACC__
	using ::max;
	using ::min;
	using ::round;
	using ::floor;
	using ::ceil;
#else
	using std::max;
	using std::min;
	using std::round;
	using std::floor;
	using std::ceil;
#endif

#ifdef HAVE_CUDA
	//define for portable function name resolution
	#if defined(__GNUC__)
	//GCC
	/// Name of the function this define is referenced. GCC version
	#define FUNCNAME_PORTABLE __PRETTY_FUNCTION__
	#elif defined(_MSC_VER)
	//Visual Studio
	/// Name of the function this define is referenced. Visual Studio version
	#define FUNCNAME_PORTABLE __FUNCSIG__
	#endif

	/// Verifies a cuda call returned "cudaSuccess". Prints error message otherwise.
	/// returns true if no error occured, false otherwise.
	#define cudaSafeCall(_err_) cudaSafeCall2(_err_, __FILE__, __LINE__, FUNCNAME_PORTABLE)

	/// Verifies a cuda call returned "cudaSuccess". Prints error message otherwise.
	/// returns true if no error occured, false otherwise. Calles by cudaSafeCall
	inline bool cudaSafeCall2(cudaError err, const char* file, int line, const char* func) {

		//#ifdef CUDA_ERROR_CHECK
		if (cudaSuccess != err) {
			char buf[1024];
			sprintf(buf, "CUDA Error (in \"%s\", Line: %d, %s): %d - %s\n", file, line, func, err, cudaGetErrorString(err));
			printf("%s", buf);
			logging::log_error(buf);
			return false;
		}

		//#endif
		return true;
	}

#ifdef HAVE_CUFFT
	/// Verifies a cuFFT call returned "CUFFT_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise.
    #define cufftSafeCall(_err_) cufftSafeCall2(_err_, __FILE__, __LINE__, FUNCNAME_PORTABLE)

	/// Verifies a cuFFT call returned "CUFFT_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise. Calles by cudaSafeCall
	inline bool cufftSafeCall2(cufftResult err, const char* file, int line, const char* func) {

		//#ifdef CUDA_ERROR_CHECK
		if (CUFFT_SUCCESS != err) {
			char buf[1024];
			sprintf(buf, "CUFFT Error (in \"%s\", Line: %d, %s): %d\n", file, line, func, err);
			printf("%s", buf);
			logging::log_error(buf);
			return false;
		}

		//#endif
		return true;
	}
#endif

	/// Returns the square of x. CUDA constexpr version
	template <typename T>
	__device__ constexpr inline T squ(const T& x)
	{
		return x*x;
	}
#else
	#define __host__
	#define __device__
#endif

	template <typename T>
	class LimitProxy
	{
	public:
		inline __host__ __device__ static T max();
		inline __host__ __device__ static T min();
	};

	template <>
	class LimitProxy<float>
	{
	public:
		inline __host__ __device__ static float max() { return FLT_MAX; }
		inline __host__ __device__ static float min() { return -FLT_MAX; }
	};

	template <>
	class LimitProxy<int16_t>
	{
	public:
		inline __host__ __device__ static int16_t max() { return 32767; }
		inline __host__ __device__ static int16_t min() { return -32767; }
	};

	template <>
	class LimitProxy<uint8_t>
	{
	public:
		inline __host__ __device__ static uint8_t max() { return 255; }
		inline __host__ __device__ static uint8_t min() { return 0; }
	};

	template <typename ResultType, typename InputType>
	__host__ __device__ ResultType clampCast(const InputType& x)
	{
		return static_cast<ResultType>(min(max(x, static_cast<InputType>(LimitProxy<ResultType>::min())), static_cast<InputType>(LimitProxy<ResultType>::max())));
	}

	template <typename ResultType, typename InputType>
	struct clampCaster {
		__host__ __device__ ResultType operator()(const InputType& a) const
		{
			return clampCast<ResultType>(a);
		}
	};
}

#endif // !__CUDAUTILITY_H__
