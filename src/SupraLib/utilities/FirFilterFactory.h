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
			createFilter(size_t length, FilterType type, FilterWindow window, double samplingFrequency, double frequency, double bandwidth = 0.0)
		{
			std::shared_ptr<Container<ElementType> > filter = createFilterNoWindow<ElementType>(length, type, samplingFrequency, frequency, bandwidth);
			applyWindowToFilter<ElementType>(filter, window);
			if (type == FilterTypeBandPass)
			{
				normalizeGain<ElementType>(filter, samplingFrequency, frequency);
			}

			return filter;
		}

	private:
		template <typename ElementType>
		static std::shared_ptr<Container<ElementType> >
			createFilterNoWindow(size_t length, FilterType type, double samplingFrequency, double frequency, double bandwidth)
		{
			ElementType omega = static_cast<ElementType>(2 * M_PI* frequency / samplingFrequency);
			ElementType omegaBandwidth = static_cast<ElementType>(2 * M_PI* bandwidth / samplingFrequency);
			int halfWidth = ((int)length - 1) / 2;

			auto filter = std::make_shared<Container<ElementType> >(LocationHost, length);

			//determine the filter function
			std::function<ElementType(int)> filterFunction = [halfWidth](int n) -> ElementType {
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
				filterFunction = [omega, halfWidth](int n) -> ElementType {
					if (n == halfWidth)
					{
						return static_cast<ElementType>(1 - omega / M_PI);
					}
					else {
						return static_cast<ElementType>(-omega / M_PI * sin(omega * (n - halfWidth)) / (omega * (n - halfWidth)));
					}
				};
				break;
			case FilterTypeBandPass:
				filterFunction = [omega, omegaBandwidth, halfWidth](int n) -> ElementType {
					if (n == halfWidth)
					{
						return static_cast<ElementType>(2.0 * omegaBandwidth / M_PI);
					}
					else {
						return static_cast<ElementType>(
							2.0 * cos(omega * n - halfWidth) *
							omegaBandwidth / M_PI * sin(omegaBandwidth * (n - halfWidth)) / (omegaBandwidth * (n - halfWidth)));
					}
				};
				break;
			case FilterTypeLowPass:
			default:
				filterFunction = [omega, halfWidth](int n) -> ElementType {
					if (n == halfWidth)
					{
						return static_cast<ElementType>(omega / M_PI);
					}
					else {
						return static_cast<ElementType>(omega / M_PI * sin(omega * (n - halfWidth)) / (omega * (n - halfWidth)));
					}
				};
				break;
			}

			//create the filter
			for (size_t k = 0; k < length; k++)
			{
				filter->get()[k] = filterFunction((int)k);
			}

			return filter;
		}

		template <typename ElementType>
		static void applyWindowToFilter(std::shared_ptr<Container<ElementType> > filter, FilterWindow window)
		{
			size_t filterLength = filter->size();
			size_t maxN = filterLength - 1;
			ElementType beta = (ElementType)4.0;
			std::function<ElementType(int)> windowFunction = [filterLength](int n) -> ElementType { return static_cast<ElementType>(1); };
			switch (window)
			{
			case FilterWindowHann:
				windowFunction = [maxN](int n) -> ElementType { return static_cast<ElementType>(
					0.50 - 0.50*cos(2 * M_PI * n / maxN)); };
				break;
			case FilterWindowHamming:
				windowFunction = [maxN](int n) -> ElementType { return static_cast<ElementType>(
					0.54 - 0.46*cos(2 * M_PI * n / maxN)); };
				break;
			case FilterWindowKaiser:
				windowFunction = [maxN, beta](int n) -> ElementType {
					double argument = beta * sqrt(1.0 - pow(2 * ((ElementType)n - maxN / 2) / maxN, 2.0));
					return static_cast<ElementType>(bessel0_1stKind(argument) / bessel0_1stKind(beta)); };
				break;
			case FilterWindowRectangular:
			default:
				windowFunction = [](int n) -> ElementType { return static_cast<ElementType>(1); };
				break;
			}

			for (size_t k = 0; k < filterLength; k++)
			{
				filter->get()[k] *= windowFunction((int)k);
			}
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
		static T bessel0_1stKind(T x)
		{
			T sum = 0.0;

			int factorial = 1;
			for (int k = 1; k < 10; k++)
			{
				T xPower = pow(x / (T)2.0, (T)k);
				factorial *= k; // like this factorial is indeed equal k!
				sum += pow(xPower / (T)factorial, (T)2.0);
			}
			return (T)1.0 + sum;
		}
	};
}

#endif // !__FIRFILTERFACTORY_H__
