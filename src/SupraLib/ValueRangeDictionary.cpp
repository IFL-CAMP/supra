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

#include "ValueRangeDictionary.h"
#include <string>

namespace supra
{
	template <>
	RangeValueType ValueRangeEntry<bool>::getType() const {
		return TypeBool;
	}
	template <>
	RangeValueType ValueRangeEntry<int8_t>::getType() const {
		return TypeInt8;
	}
	template <>
	RangeValueType ValueRangeEntry<uint8_t>::getType() const {
		return TypeUint8;
	}
	template <>
	RangeValueType ValueRangeEntry<int16_t>::getType() const {
		return TypeInt16;
	}
	template <>
	RangeValueType ValueRangeEntry<uint16_t>::getType() const {
		return TypeUint16;
	}
	template <>
	RangeValueType ValueRangeEntry<int32_t>::getType() const {
		return TypeInt32;
	}
	template <>
	RangeValueType ValueRangeEntry<uint32_t>::getType() const {
		return TypeUint32;
	}
	template <>
	RangeValueType ValueRangeEntry<int64_t>::getType() const {
		return TypeInt64;
	}
	template <>
	RangeValueType ValueRangeEntry<uint64_t>::getType() const {
		return TypeUint64;
	}
	template <>
	RangeValueType ValueRangeEntry<float>::getType() const {
		return TypeFloat;
	}
	template <>
	RangeValueType ValueRangeEntry<double>::getType() const {
		return TypeDouble;
	}
	template <>
	RangeValueType ValueRangeEntry<std::string>::getType() const {
		return TypeString;
	}
}