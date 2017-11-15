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

#include <string>
#include <stdint.h>

#include "TemplateTypeDefault.h"

namespace supra
{
	/// Returns the default value for this type
	template <>
	bool TemplateTypeDefault<bool>::getDefault()
	{
		return false;
	}
	/// Returns the default value for this type
	template <>
	int8_t TemplateTypeDefault<int8_t>::getDefault()
	{
		return 0;
	}
	/// Returns the default value for this type
	template <>
	uint8_t TemplateTypeDefault<uint8_t>::getDefault()
	{
		return 0;
	}
	/// Returns the default value for this type
	template <>
	int16_t TemplateTypeDefault<int16_t>::getDefault()
	{
		return 0;
	}
	/// Returns the default value for this type
	template <>
	uint16_t TemplateTypeDefault<uint16_t>::getDefault()
	{
		return 0;
	}
	/// Returns the default value for this type
	template <>
	int32_t TemplateTypeDefault<int32_t>::getDefault()
	{
		return 0;
	}
	/// Returns the default value for this type
	template <>
	uint32_t TemplateTypeDefault<uint32_t>::getDefault()
	{
		return 0;
	}
	/// Returns the default value for this type
	template <>
	int64_t TemplateTypeDefault<int64_t>::getDefault()
	{
		return 0;
	}
	/// Returns the default value for this type
	template <>
	uint64_t TemplateTypeDefault<uint64_t>::getDefault()
	{
		return 0;
	}
	/// Returns the default value for this type
	template <>
	float TemplateTypeDefault<float>::getDefault()
	{
		return 0.0f;
	}
	/// Returns the default value for this type
	template <>
	double TemplateTypeDefault<double>::getDefault()
	{
		return 0.0;
	}
	/// Returns the default value for this type
	template <>
	std::string TemplateTypeDefault<std::string>::getDefault()
	{
		return "";
	}
}