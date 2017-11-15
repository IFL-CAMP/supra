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

#ifndef __COMMANDLINEINTERFACE_H__
#define __COMMANDLINEINTERFACE_H__

#include <string>
#include <ValueRangeDictionary.h>
#include <ConfigurationDictionary.h>
#include <AbstractInput.h>

#include <utilities/utility.h>

namespace supra
{
	using std::to_string;

	/// The SUPRA command line interface
	class CommandlineInterface
	{
	public:
		/// Entry-point for the interface
		void mainMenu();
		/// Sub-menu for modification of an input node
		void editInputNode(unsigned int nodeNum);
		/// Sub-menu for modification of an output node
		void editOutputNode(unsigned int nodeNum);
		/// Sub-menu for modification of an general node
		void editNode(unsigned int nodeNum);

	private:
		std::string getParameterRangeString(const ValueRangeDictionary* ranges, std::string paramName);
		void readAndSetParameter(std::shared_ptr<AbstractInput<RecordObject> > inputNode, std::string paramName);
		std::string getParameterTypeString(const ValueRangeDictionary* ranges, std::string paramName);
		std::string getParameterValueString(const ConfigurationDictionary* config, const ValueRangeDictionary* ranges, std::string paramName);

		template <typename ValueType>
		std::string getParameterRangeStringTemplated(const ValueRangeDictionary* ranges, std::string paramName)
		{
			using ::std::to_string;
			auto range = ranges->get<ValueType>(paramName);
			std::string retVal;
			if (range->isUnrestricted())
			{
				retVal = "*";
			}
			else if (range->isContinuous())
			{
				retVal = "[ " + to_string(range->getContinuous().first) + " , " +
					to_string(range->getContinuous().second) + " ]";
			}
			else
			{
				auto valueRange = range->getDiscrete();
				retVal = "{ ";
				for (size_t i = 0; i < valueRange.size(); i++)
				{
					retVal += to_string(valueRange[i]);
					if (i < valueRange.size() - 1)
					{
						retVal += " , ";
					}
				}
				retVal += " }";
			}
			return retVal;
		}
	};
}
#endif //!__COMMANDLINEINTERFACE_H__
