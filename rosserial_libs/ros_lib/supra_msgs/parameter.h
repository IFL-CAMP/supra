#ifndef _ROS_supra_msgs_parameter_h
#define _ROS_supra_msgs_parameter_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace supra_msgs
{

  class parameter : public ros::Msg
  {
    public:
      typedef const char* _parameterId_type;
      _parameterId_type parameterId;
      typedef const char* _displayName_type;
      _displayName_type displayName;
      typedef const char* _type_type;
      _type_type type;
      typedef const char* _value_type;
      _value_type value;
      typedef int16_t _rangeType_type;
      _rangeType_type rangeType;
      typedef const char* _rangeStart_type;
      _rangeStart_type rangeStart;
      typedef const char* _rangeEnd_type;
      _rangeEnd_type rangeEnd;
      uint32_t discreteValues_length;
      typedef char* _discreteValues_type;
      _discreteValues_type st_discreteValues;
      _discreteValues_type * discreteValues;
      enum { rangeTypeUnrestricted = 0 };
      enum { rangeTypeContinuous = 1 };
      enum { rangeTypeDiscrete = 2 };

    parameter():
      parameterId(""),
      displayName(""),
      type(""),
      value(""),
      rangeType(0),
      rangeStart(""),
      rangeEnd(""),
      discreteValues_length(0), discreteValues(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_parameterId = strlen(this->parameterId);
      varToArr(outbuffer + offset, length_parameterId);
      offset += 4;
      memcpy(outbuffer + offset, this->parameterId, length_parameterId);
      offset += length_parameterId;
      uint32_t length_displayName = strlen(this->displayName);
      varToArr(outbuffer + offset, length_displayName);
      offset += 4;
      memcpy(outbuffer + offset, this->displayName, length_displayName);
      offset += length_displayName;
      uint32_t length_type = strlen(this->type);
      varToArr(outbuffer + offset, length_type);
      offset += 4;
      memcpy(outbuffer + offset, this->type, length_type);
      offset += length_type;
      uint32_t length_value = strlen(this->value);
      varToArr(outbuffer + offset, length_value);
      offset += 4;
      memcpy(outbuffer + offset, this->value, length_value);
      offset += length_value;
      union {
        int16_t real;
        uint16_t base;
      } u_rangeType;
      u_rangeType.real = this->rangeType;
      *(outbuffer + offset + 0) = (u_rangeType.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_rangeType.base >> (8 * 1)) & 0xFF;
      offset += sizeof(this->rangeType);
      uint32_t length_rangeStart = strlen(this->rangeStart);
      varToArr(outbuffer + offset, length_rangeStart);
      offset += 4;
      memcpy(outbuffer + offset, this->rangeStart, length_rangeStart);
      offset += length_rangeStart;
      uint32_t length_rangeEnd = strlen(this->rangeEnd);
      varToArr(outbuffer + offset, length_rangeEnd);
      offset += 4;
      memcpy(outbuffer + offset, this->rangeEnd, length_rangeEnd);
      offset += length_rangeEnd;
      *(outbuffer + offset + 0) = (this->discreteValues_length >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->discreteValues_length >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->discreteValues_length >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->discreteValues_length >> (8 * 3)) & 0xFF;
      offset += sizeof(this->discreteValues_length);
      for( uint32_t i = 0; i < discreteValues_length; i++){
      uint32_t length_discreteValuesi = strlen(this->discreteValues[i]);
      varToArr(outbuffer + offset, length_discreteValuesi);
      offset += 4;
      memcpy(outbuffer + offset, this->discreteValues[i], length_discreteValuesi);
      offset += length_discreteValuesi;
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_parameterId;
      arrToVar(length_parameterId, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_parameterId; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_parameterId-1]=0;
      this->parameterId = (char *)(inbuffer + offset-1);
      offset += length_parameterId;
      uint32_t length_displayName;
      arrToVar(length_displayName, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_displayName; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_displayName-1]=0;
      this->displayName = (char *)(inbuffer + offset-1);
      offset += length_displayName;
      uint32_t length_type;
      arrToVar(length_type, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_type; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_type-1]=0;
      this->type = (char *)(inbuffer + offset-1);
      offset += length_type;
      uint32_t length_value;
      arrToVar(length_value, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_value; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_value-1]=0;
      this->value = (char *)(inbuffer + offset-1);
      offset += length_value;
      union {
        int16_t real;
        uint16_t base;
      } u_rangeType;
      u_rangeType.base = 0;
      u_rangeType.base |= ((uint16_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_rangeType.base |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->rangeType = u_rangeType.real;
      offset += sizeof(this->rangeType);
      uint32_t length_rangeStart;
      arrToVar(length_rangeStart, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_rangeStart; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_rangeStart-1]=0;
      this->rangeStart = (char *)(inbuffer + offset-1);
      offset += length_rangeStart;
      uint32_t length_rangeEnd;
      arrToVar(length_rangeEnd, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_rangeEnd; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_rangeEnd-1]=0;
      this->rangeEnd = (char *)(inbuffer + offset-1);
      offset += length_rangeEnd;
      uint32_t discreteValues_lengthT = ((uint32_t) (*(inbuffer + offset))); 
      discreteValues_lengthT |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1); 
      discreteValues_lengthT |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2); 
      discreteValues_lengthT |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3); 
      offset += sizeof(this->discreteValues_length);
      if(discreteValues_lengthT > discreteValues_length)
        this->discreteValues = (char**)realloc(this->discreteValues, discreteValues_lengthT * sizeof(char*));
      discreteValues_length = discreteValues_lengthT;
      for( uint32_t i = 0; i < discreteValues_length; i++){
      uint32_t length_st_discreteValues;
      arrToVar(length_st_discreteValues, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_discreteValues; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_discreteValues-1]=0;
      this->st_discreteValues = (char *)(inbuffer + offset-1);
      offset += length_st_discreteValues;
        memcpy( &(this->discreteValues[i]), &(this->st_discreteValues), sizeof(char*));
      }
     return offset;
    }

    const char * getType(){ return "supra_msgs/parameter"; };
    const char * getMD5(){ return "bcd42149fbfad5be634c401874dd64a1"; };

  };

}
#endif