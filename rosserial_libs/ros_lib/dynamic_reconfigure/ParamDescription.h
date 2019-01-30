#ifndef _ROS_dynamic_reconfigure_ParamDescription_h
#define _ROS_dynamic_reconfigure_ParamDescription_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace dynamic_reconfigure
{

  class ParamDescription : public ros::Msg
  {
    public:
      const char* name;
      const char* type;
      uint32_t level;
      const char* description;
      const char* edit_method;

    ParamDescription():
      name(""),
      type(""),
      level(0),
      description(""),
      edit_method("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_name = strlen(this->name);
      memcpy(outbuffer + offset, &length_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->name, length_name);
      offset += length_name;
      uint32_t length_type = strlen(this->type);
      memcpy(outbuffer + offset, &length_type, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->type, length_type);
      offset += length_type;
      *(outbuffer + offset + 0) = (this->level >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->level >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->level >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->level >> (8 * 3)) & 0xFF;
      offset += sizeof(this->level);
      uint32_t length_description = strlen(this->description);
      memcpy(outbuffer + offset, &length_description, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->description, length_description);
      offset += length_description;
      uint32_t length_edit_method = strlen(this->edit_method);
      memcpy(outbuffer + offset, &length_edit_method, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->edit_method, length_edit_method);
      offset += length_edit_method;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_name;
      memcpy(&length_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_name-1]=0;
      this->name = (char *)(inbuffer + offset-1);
      offset += length_name;
      uint32_t length_type;
      memcpy(&length_type, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_type; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_type-1]=0;
      this->type = (char *)(inbuffer + offset-1);
      offset += length_type;
      this->level =  ((uint32_t) (*(inbuffer + offset)));
      this->level |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->level |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->level |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->level);
      uint32_t length_description;
      memcpy(&length_description, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_description; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_description-1]=0;
      this->description = (char *)(inbuffer + offset-1);
      offset += length_description;
      uint32_t length_edit_method;
      memcpy(&length_edit_method, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_edit_method; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_edit_method-1]=0;
      this->edit_method = (char *)(inbuffer + offset-1);
      offset += length_edit_method;
     return offset;
    }

    const char * getType(){ return "dynamic_reconfigure/ParamDescription"; };
    const char * getMD5(){ return "7434fcb9348c13054e0c3b267c8cb34d"; };

  };

}
#endif