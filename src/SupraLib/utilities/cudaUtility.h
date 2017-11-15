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

#include <cuda_runtime_api.h>
#include <cstdio>
#include "utilities/Logging.h"
#include <algorithm>
#include <cmath>

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

	/// Returns the square of x. CUDA constexpr version
	template <typename T>
	__device__ constexpr inline T squ(const T& x)
	{
		return x*x;
	}
}

#endif // !__CUDAUTILITY_H__
