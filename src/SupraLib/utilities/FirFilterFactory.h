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

#ifndef __FIRFILTERFACTORY_H__
#define __FIRFILTERFACTORY_H__

#include <memory>
#include <functional>
#include "Container.h"

namespace supra
{
	/// A factory for FIR filters
	class FirFilterFactory {
	public:
		/// Enum for the different filter types
		enum FilterType {
			FilterTypeLowPass,
			FilterTypeHighPass,
			FilterTypeBandPass
		};

		/// Enum for the different window types used in creating filters
		enum FilterWindow {
			FilterWindowRectangular,
			FilterWindowHann,
			FilterWindowHamming,
			FilterWindowKaiser
		};

		/// Returns a FIR filter constructed with the window-method
		template <typename ElementType>
		static std::shared_ptr<Container<ElementType> >
			createFilter(const size_t &length, const FilterType &type, const FilterWindow &window, const double &samplingFrequency, const double &frequency, const double &bandwidth = 0.0)
		{
			std::shared_ptr<Container<ElementType> > filter = createFilterNoWindow<ElementType>(length, type, samplingFrequency, frequency, bandwidth);			
			applyWindowToFilter<ElementType>(filter, window);
			if (type == FilterTypeBandPass)
				normalizeGain<ElementType>(filter, samplingFrequency, frequency);
			return filter;
		}

	private:

		template <typename ElementType>
		static std::shared_ptr<Container<ElementType> >
			createFilterNoWindow(const size_t &length, const FilterType &type, const double &samplingFrequency, const double &frequency, const double &bandwidth)
		{
			ElementType omega = static_cast<ElementType>(2 * M_PI* frequency / samplingFrequency);
			ElementType omegaBandwidth = static_cast<ElementType>(2 * M_PI* bandwidth / samplingFrequency);
			ElementType omegaByMPI = static_cast<ElementType>(omega / M_PI);
			ElementType omegaBandByMPI = static_cast<ElementType>(omegaBandwidth / M_PI);
			int halfWidth = ((int)length - 1) / 2;

			auto filter = std::make_shared<Container<ElementType> >(LocationHost, cudaStreamPerThread, length);

			//determine the filter function
			std::function<ElementType(int)> filterFunction = [&halfWidth](int n) -> ElementType {
				if (n == halfWidth)
				{
					return static_cast<ElementType>(1);
				}
				else {
					return static_cast<ElementType>(0);
				}
			};
			switch (type)
			{
				case FilterTypeHighPass:
					filterFunction = [&omega, &halfWidth, &omegaByMPI](int n) -> ElementType {
						if (n == halfWidth)
						{
							return static_cast<ElementType>(1 - omegaByMPI);
						}
						else {
							return static_cast<ElementType>(-omegaByMPI * sin(omega * (n - halfWidth)) / (omega * (n - halfWidth)));
						}
					};
					break;
				case FilterTypeBandPass:
					filterFunction = [&omega, &omegaBandwidth, &halfWidth, &omegaBandByMPI](int n) -> ElementType {
						if (n == halfWidth)
						{
							return static_cast<ElementType>(2.0 * omegaBandByMPI);
						}
						else {
							return static_cast<ElementType>(
								2.0 * cos(omega * n - halfWidth) *
								omegaBandByMPI * sin(omegaBandwidth * (n - halfWidth)) / (omegaBandwidth * (n - halfWidth)));
						}
					};
					break;
				case FilterTypeLowPass:
				default:
					filterFunction = [&omega, &halfWidth, &omegaByMPI](int n) -> ElementType {
						if (n == halfWidth)
						{
							return static_cast<ElementType>(omegaByMPI);
						}
						else {
							return static_cast<ElementType>(omegaByMPI * sin(omega * (n - halfWidth)) / (omega * (n - halfWidth)));
						}
					};
					break;
			}

			//create the filter
			for (size_t k = 0; k < length; k++) {
				filter->get()[k] = filterFunction((int)k);
			}
			return filter;
		}

		template <typename ElementType>
		static void applyWindowToFilter(std::shared_ptr<Container<ElementType> > filter, FilterWindow window)
		{
			size_t filterLength = filter->size();
			size_t maxN = filterLength - 1;
			size_t maxNby2 = maxN / 2;
			ElementType beta = (ElementType)4.0;
			ElementType mpi2bymaxN = (ElementType)(2.0 * M_PI / maxN);
			std::function<ElementType(int)> windowFunction = [filterLength](int n) -> ElementType { return static_cast<ElementType>(1); };
			switch (window)
			{
			case FilterWindowHann:
				windowFunction = [&mpi2bymaxN](int n) -> ElementType { return static_cast<ElementType>(
					0.50 - 0.50*cos(mpi2bymaxN * n)); };
				break;
			case FilterWindowHamming:
				windowFunction = [&mpi2bymaxN](int n) -> ElementType { return static_cast<ElementType>(
					0.54 - 0.46*cos(mpi2bymaxN * n)); };
				break;
			case FilterWindowKaiser:
				windowFunction = [&maxNby2, &beta](int n) -> ElementType {
					double argument = beta * sqrt(1.0 - pow(((ElementType)n - maxNby2) / maxNby2, 2.0));
					return static_cast<ElementType>(bessel0_1stKind(argument) / bessel0_1stKind(beta)); };
				break;
			case FilterWindowRectangular:
			default:
				windowFunction = [](int n) -> ElementType { return static_cast<ElementType>(1); };
				break;
			}

			for (size_t k = 0; k < filterLength; k++)
				filter->get()[k] *= windowFunction((int)k);
		}

		template <typename ElementType>
		static void normalizeGain(std::shared_ptr<Container<ElementType> > filter, double samplingFrequency, double frequency)
		{
			ElementType omega = static_cast<ElementType>(2 * M_PI* frequency / samplingFrequency);
			ElementType gainR = 0;
			ElementType gainI = 0;

			for (int k = 0; k < filter->size(); k++)
			{
				gainR += filter->get()[k] * cos(omega * (ElementType)k);
				gainI += filter->get()[k] * sin(omega * (ElementType)k);
			}
			ElementType gain = sqrt(gainR*gainR + gainI*gainI);
			for (int k = 0; k < filter->size(); k++)
			{
				filter->get()[k] /= gain;
			}
		}

		template <typename T>
		static T bessel0_1stKind(const T &x)
		{
			T sum = 0.0;
			//implemented look up factorial. 
			static const int factorial[9] = { 1, 2, 6, 24, 120, 720, 5040, 40320, 362880 };
			//int factorial = 1;
			for (int k = 1; k < 10; k++)
			{
				T xPower = pow(x / (T)2.0, (T)k);
				//factorial *= k; // like this factorial is indeed equal k!
				// 1, 2, 6, 24, 120, 720, 5040, 40320, 362880
				//sum += pow(xPower / (T)factorial, (T)2.0);
				sum += pow(xPower / (T)factorial[k-1], (T)2.0);
			}
			return (T)1.0 + sum;
		}
	};
}

#endif // !__FIRFILTERFACTORY_H__
