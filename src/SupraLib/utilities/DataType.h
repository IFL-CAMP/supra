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

#ifndef __DATATYPE_H__
#define __DATATYPE_H__

#include <stdint.h>
#include <string>
#include "utilities/utility.h"

namespace supra
{
	/// Enum for the types used in containers and by the parameter system
	enum DataType
	{
		TypeBool,
		TypeInt8,
		TypeUint8,
		TypeInt16,
		TypeUint16,
		TypeInt32,
		TypeUint32,
		TypeInt64,
		TypeUint64,
		TypeFloat,
		TypeDouble,
		TypeString,
		TypeDataType,
		TypeUnknown
	};

	template <typename T>
	DataType DataTypeGet()
	{
		return TypeUnknown;
	}

	template <>
	DataType DataTypeGet<bool>();
	template <>
	DataType DataTypeGet<int8_t>();
	template <>
	DataType DataTypeGet<uint8_t>();
	template <>
	DataType DataTypeGet<int16_t>();
	template <>
	DataType DataTypeGet<uint16_t>();
	template <>
	DataType DataTypeGet<int32_t>();
	template <>
	DataType DataTypeGet<uint32_t>();
	template <>
	DataType DataTypeGet<int64_t>();
	template <>
	DataType DataTypeGet<uint64_t>();
	template <>
	DataType DataTypeGet<float>();
	template <>
	DataType DataTypeGet<double>();
	template <>
	DataType DataTypeGet<std::string>();
	template <>
	DataType DataTypeGet<DataType>();

	std::ostream& operator<<(std::ostream& os, DataType dataType);
	std::istream& operator>>(std::istream& is, DataType& dataType);
}

#endif // !__DATATYPE_H__
