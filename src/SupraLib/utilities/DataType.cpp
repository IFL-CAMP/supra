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

#include "DataType.h"

namespace supra
{
	template <>
	DataType DataTypeGet<bool>() {
		return TypeBool;
	}
	template <>
	DataType DataTypeGet<int8_t>() {
		return TypeInt8;
	}
	template <>
	DataType DataTypeGet<uint8_t>() {
		return TypeUint8;
	}
	template <>
	DataType DataTypeGet<int16_t>() {
		return TypeInt16;
	}
	template <>
	DataType DataTypeGet<uint16_t>() {
		return TypeUint16;
	}
	template <>
	DataType DataTypeGet<int32_t>() {
		return TypeInt32;
	}
	template <>
	DataType DataTypeGet<uint32_t>() {
		return TypeUint32;
	}
	template <>
	DataType DataTypeGet<int64_t>() {
		return TypeInt64;
	}
	template <>
	DataType DataTypeGet<uint64_t>() {
		return TypeUint64;
	}
	template <>
	DataType DataTypeGet<float>() {
		return TypeFloat;
	}
	template <>
	DataType DataTypeGet<double>() {
		return TypeDouble;
	}
	template <>
	DataType DataTypeGet<std::string>() {
		return TypeString;
	}

	template <>
	DataType DataTypeGet<DataType>() {
		return TypeDataType;
	}

	std::ostream& operator<<(std::ostream& os, DataType dataType)
	{
		switch (dataType)
		{
		case supra::TypeBool:
			os << "bool";
			break;
		case supra::TypeInt8:
			os << "int8";
			break;
		case supra::TypeUint8:
			os << "uint8";
			break;
		case supra::TypeInt16:
			os << "int16";
			break;
		case supra::TypeUint16:
			os << "uint16";
			break;
		case supra::TypeInt32:
			os << "int32";
			break;
		case supra::TypeUint32:
			os << "uint32";
			break;
		case supra::TypeInt64:
			os << "int64";
			break;
		case supra::TypeUint64:
			os << "uint64";
			break;
		case supra::TypeFloat:
			os << "float";
			break;
		case supra::TypeDouble:
			os << "double";
			break;
		case supra::TypeString:
			os << "string";
			break;
		case supra::TypeDataType:
			os << "dataType";
			break;
		case supra::TypeUnknown:
			os << "Unknown";
			break;
		default:
			os.setstate(std::ios_base::failbit);
			break;
		}
		
		return os;
	}

	std::istream& operator>>(std::istream& is, DataType& dataType)
	{
		std::string s;
		is >> s;

		if (s == "bool")
		{
			dataType = supra::TypeBool;
		}
		else if (s == "int8")
		{
			dataType = supra::TypeInt8;
		}
		else if (s == "uint8")
		{
			dataType = supra::TypeUint8;
		}
		else if (s == "int16")
		{
			dataType = supra::TypeInt16;
		}
		else if (s == "uint16")
		{
			dataType = supra::TypeUint16;
		}
		else if (s == "int32")
		{
			dataType = supra::TypeInt32;
		}
		else if (s == "uint32")
		{
			dataType = supra::TypeUint32;
		}
		else if (s == "int64")
		{
			dataType = supra::TypeInt64;
		}
		else if (s == "uint64")
		{
			dataType = supra::TypeUint64;
		}
		else if (s == "float")
		{
			dataType = supra::TypeFloat;
		}
		else if (s == "double")
		{
			dataType = supra::TypeDouble;
		}
		else if (s == "string")
		{
			dataType = supra::TypeString;
		}
		else if (s == "dataType")
		{
			dataType = supra::TypeDataType;
		}
		else if (s == "Unknown")
		{
			dataType = supra::TypeUnknown;
		}
		else
		{
			is.setstate(std::ios_base::failbit);
		}
		return is;
	}
}
