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

#ifndef __CUBLASUTILITY_H__
#define __CUBLASUTILITY_H__

#ifdef HAVE_CUDA_CUBLAS
#include <cublas_v2.h>
#endif

namespace supra
{

#ifdef HAVE_CUDA_CUBLAS
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
#endif //HAVE_CUDA_CUBLAS
}

#endif // !__CUBLASUTILITY_H__
