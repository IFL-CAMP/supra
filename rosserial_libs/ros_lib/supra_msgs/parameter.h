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
      const char* parameterId;
      const char* displayName;
      const char* type;
      const char* value;
      int16_t rangeType;
      const char* rangeStart;
      const char* rangeEnd;
      uint8_t discreteValues_length;
      char* st_discreteValues;
      char* * discreteValues;
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
      memcpy(outbuffer + offset, &length_parameterId, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->parameterId, length_parameterId);
      offset += length_parameterId;
      uint32_t length_displayName = strlen(this->displayName);
      memcpy(outbuffer + offset, &length_displayName, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->displayName, length_displayName);
      offset += length_displayName;
      uint32_t length_type = strlen(this->type);
      memcpy(outbuffer + offset, &length_type, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->type, length_type);
      offset += length_type;
      uint32_t length_value = strlen(this->value);
      memcpy(outbuffer + offset, &length_value, sizeof(uint32_t));
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
      memcpy(outbuffer + offset, &length_rangeStart, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->rangeStart, length_rangeStart);
      offset += length_rangeStart;
      uint32_t length_rangeEnd = strlen(this->rangeEnd);
      memcpy(outbuffer + offset, &length_rangeEnd, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->rangeEnd, length_rangeEnd);
      offset += length_rangeEnd;
      *(outbuffer + offset++) = discreteValues_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < discreteValues_length; i++){
      uint32_t length_discreteValuesi = strlen(this->discreteValues[i]);
      memcpy(outbuffer + offset, &length_discreteValuesi, sizeof(uint32_t));
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
      memcpy(&length_parameterId, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_parameterId; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_parameterId-1]=0;
      this->parameterId = (char *)(inbuffer + offset-1);
      offset += length_parameterId;
      uint32_t length_displayName;
      memcpy(&length_displayName, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_displayName; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_displayName-1]=0;
      this->displayName = (char *)(inbuffer + offset-1);
      offset += length_displayName;
      uint32_t length_type;
      memcpy(&length_type, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_type; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_type-1]=0;
      this->type = (char *)(inbuffer + offset-1);
      offset += length_type;
      uint32_t length_value;
      memcpy(&length_value, (inbuffer + offset), sizeof(uint32_t));
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
      memcpy(&length_rangeStart, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_rangeStart; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_rangeStart-1]=0;
      this->rangeStart = (char *)(inbuffer + offset-1);
      offset += length_rangeStart;
      uint32_t length_rangeEnd;
      memcpy(&length_rangeEnd, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_rangeEnd; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_rangeEnd-1]=0;
      this->rangeEnd = (char *)(inbuffer + offset-1);
      offset += length_rangeEnd;
      uint8_t discreteValues_lengthT = *(inbuffer + offset++);
      if(discreteValues_lengthT > discreteValues_length)
        this->discreteValues = (char**)realloc(this->discreteValues, discreteValues_lengthT * sizeof(char*));
      offset += 3;
      discreteValues_length = discreteValues_lengthT;
      for( uint8_t i = 0; i < discreteValues_length; i++){
      uint32_t length_st_discreteValues;
      memcpy(&length_st_discreteValues, (inbuffer + offset), sizeof(uint32_t));
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