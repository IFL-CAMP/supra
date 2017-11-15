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

#include "LogCompressor.h"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

using namespace std;

namespace supra
{
	template <typename In, typename Out, typename WorkType>
	struct thrustLogcompress : public thrust::unary_function<In, Out>
	{
		WorkType _outMax;
		WorkType _inScale;
		WorkType _scaleOverDenominator;

		// Thrust functor that computes
		// signal = log10(1 + a*signal)./log10(1 + a) 
		// of the downscaled (_inMax) input signal
		thrustLogcompress(double dynamicRange, In inMax, Out outMax, double scale)
			: _outMax(static_cast<WorkType>(outMax))
			, _inScale(static_cast<WorkType>(dynamicRange / inMax))
			, _scaleOverDenominator(static_cast<WorkType>(scale * outMax / log10(dynamicRange + 1)))
		{};

		__host__ __device__ Out operator()(const In& a) const
		{
			WorkType val = min(log10(abs(static_cast<WorkType>(a))*_inScale + 1) * _scaleOverDenominator, _outMax);
			return static_cast<Out>(val);
		}
	};

	template<>
	shared_ptr<Container<uint8_t> > LogCompressor::compress(const shared_ptr<const Container<int16_t>>& inImageData, vec3s size, double dynamicRange, double scale, double inMax)
	{
		size_t width = size.x;
		size_t height = size.y;
		size_t depth = size.z;

		auto pComprGpu = make_shared<Container<uint8_t> >(LocationGpu, width*height*depth);

		thrustLogcompress<int16_t, uint8_t, WorkType> c(pow(10, (dynamicRange / 20)), static_cast<int16_t>(inMax), std::numeric_limits<uint8_t>::max(), scale);
		thrust::transform(thrust::device, inImageData->get(), inImageData->get() + (width*height*depth),
			pComprGpu->get(), c);
		cudaSafeCall(cudaPeekAtLastError());
		cudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));

		return pComprGpu;
	}
}