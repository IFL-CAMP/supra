#ifndef _ROS_SERVICE_ListControllerTypes_h
#define _ROS_SERVICE_ListControllerTypes_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace controller_manager_msgs
{

static const char LISTCONTROLLERTYPES[] = "controller_manager_msgs/ListControllerTypes";

  class ListControllerTypesRequest : public ros::Msg
  {
    public:

    ListControllerTypesRequest()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
     return offset;
    }

    const char * getType(){ return LISTCONTROLLERTYPES; };
    const char * getMD5(){ return "d41d8cd98f00b204e9800998ecf8427e"; };

  };

  class ListControllerTypesResponse : public ros::Msg
  {
    public:
      uint8_t types_length;
      char* st_types;
      char* * types;
      uint8_t base_classes_length;
      char* st_base_classes;
      char* * base_classes;

    ListControllerTypesResponse():
      types_length(0), types(NULL),
      base_classes_length(0), base_classes(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = types_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < types_length; i++){
      uint32_t length_typesi = strlen(this->types[i]);
      memcpy(outbuffer + offset, &length_typesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->types[i], length_typesi);
      offset += length_typesi;
      }
      *(outbuffer + offset++) = base_classes_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < base_classes_length; i++){
      uint32_t length_base_classesi = strlen(this->base_classes[i]);
      memcpy(outbuffer + offset, &length_base_classesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->base_classes[i], length_base_classesi);
      offset += length_base_classesi;
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t types_lengthT = *(inbuffer + offset++);
      if(types_lengthT > types_length)
        this->types = (char**)realloc(this->types, types_lengthT * sizeof(char*));
      offset += 3;
      types_length = types_lengthT;
      for( uint8_t i = 0; i < types_length; i++){
      uint32_t length_st_types;
      memcpy(&length_st_types, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_types; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_types-1]=0;
      this->st_types = (char *)(inbuffer + offset-1);
      offset += length_st_types;
        memcpy( &(this->types[i]), &(this->st_types), sizeof(char*));
      }
      uint8_t base_classes_lengthT = *(inbuffer + offset++);
      if(base_classes_lengthT > base_classes_length)
        this->base_classes = (char**)realloc(this->base_classes, base_classes_lengthT * sizeof(char*));
      offset += 3;
      base_classes_length = base_classes_lengthT;
      for( uint8_t i = 0; i < base_classes_length; i++){
      uint32_t length_st_base_classes;
      memcpy(&length_st_base_classes, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_base_classes; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_base_classes-1]=0;
      this->st_base_classes = (char *)(inbuffer + offset-1);
      offset += length_st_base_classes;
        memcpy( &(this->base_classes[i]), &(this->st_base_classes), sizeof(char*));
      }
     return offset;
    }

    const char * getType(){ return LISTCONTROLLERTYPES; };
    const char * getMD5(){ return "c1d4cd11aefa9f97ba4aeb5b33987f4e"; };

  };

  class ListControllerTypes {
    public:
    typedef ListControllerTypesRequest Request;
    typedef ListControllerTypesResponse Response;
  };

}
#endif
