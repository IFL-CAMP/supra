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
#ifdef HAVE_CUDA
	template <>
	DataType DataTypeGet<__half>() {
		return TypeHalf;
	}
#endif
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

	DataType DataTypeFromString(const std::string & s, bool* success)
	{
		DataType dataType;
		bool hadSuccess = true;
		if (s == "bool")
		{
			dataType = supra::TypeBool;
		}
		else if (s == "int8" || s == "int8_t")
		{
			dataType = supra::TypeInt8;
		}
		else if (s == "uint8" || s == "uint8_t")
		{
			dataType = supra::TypeUint8;
		}
		else if (s == "int16" || s == "int16_t")
		{
			dataType = supra::TypeInt16;
		}
		else if (s == "uint16" || s == "uint16_t")
		{
			dataType = supra::TypeUint16;
		}
		else if (s == "int32" || s == "int32_t")
		{
			dataType = supra::TypeInt32;
		}
		else if (s == "uint32" || s == "uint32_t")
		{
			dataType = supra::TypeUint32;
		}
		else if (s == "int64" || s == "int64_t")
		{
			dataType = supra::TypeInt64;
		}
		else if (s == "uint64" || s == "uint64_t")
		{
			dataType = supra::TypeUint64;
		}
#ifdef HAVE_CUDA
		else if (s == "half")
		{
			dataType = supra::TypeHalf;
		}
#endif
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
			hadSuccess = false;
		}
		if (success)
		{
			*success = hadSuccess;
		}
		return dataType;
	}

	std::string DataTypeToString(DataType t, bool* success)
	{
		std::string s;
		bool hadSuccess = true;
		switch (t)
		{
		case supra::TypeBool:
			s = "bool";
			break;
		case supra::TypeInt8:
			s = "int8_t";
			break;
		case supra::TypeUint8:
			s = "uint8_t";
			break;
		case supra::TypeInt16:
			s = "int16_t";
			break;
		case supra::TypeUint16:
			s = "uint16_t";
			break;
		case supra::TypeInt32:
			s = "int32_t";
			break;
		case supra::TypeUint32:
			s = "uint32_t";
			break;
		case supra::TypeInt64:
			s = "int64_t";
			break;
		case supra::TypeUint64:
			s = "uint64_t";
			break;
#ifdef HAVE_CUDA
		case supra::TypeHalf:
			s = "half";
			break;
#endif
		case supra::TypeFloat:
			s = "float";
			break;
		case supra::TypeDouble:
			s = "double";
			break;
		case supra::TypeString:
			s = "string";
			break;
		case supra::TypeDataType:
			s = "DataType";
			break;
		case supra::TypeUnknown:
			s = "Unknown";
			break;
		default:
			hadSuccess = false;
			break;
		}

		if (success)
		{
			*success = hadSuccess;
		}
		return s;
	}

	std::ostream& operator<<(std::ostream& os, DataType dataType)
	{
		bool success;
		std::string s = DataTypeToString(dataType, &success);

		if (success)
		{
			os << s;
		}
		else
		{
			os.setstate(std::ios_base::failbit);
		}

		return os;
	}

	std::istream& operator>>(std::istream& is, DataType& dataType)
	{
		std::string s;
		is >> s;

		bool success;
		dataType = DataTypeFromString(s, &success);
		if(!success)
		{
			is.setstate(std::ios_base::failbit);
		}
		return is;
	}
}
