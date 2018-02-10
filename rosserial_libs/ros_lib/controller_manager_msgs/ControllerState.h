#ifndef _ROS_controller_manager_msgs_ControllerState_h
#define _ROS_controller_manager_msgs_ControllerState_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace controller_manager_msgs
{

  class ControllerState : public ros::Msg
  {
    public:
      const char* name;
      const char* state;
      const char* type;
      const char* hardware_interface;
      uint8_t resources_length;
      char* st_resources;
      char* * resources;

    ControllerState():
      name(""),
      state(""),
      type(""),
      hardware_interface(""),
      resources_length(0), resources(NULL)
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
      uint32_t length_state = strlen(this->state);
      memcpy(outbuffer + offset, &length_state, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->state, length_state);
      offset += length_state;
      uint32_t length_type = strlen(this->type);
      memcpy(outbuffer + offset, &length_type, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->type, length_type);
      offset += length_type;
      uint32_t length_hardware_interface = strlen(this->hardware_interface);
      memcpy(outbuffer + offset, &length_hardware_interface, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->hardware_interface, length_hardware_interface);
      offset += length_hardware_interface;
      *(outbuffer + offset++) = resources_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < resources_length; i++){
      uint32_t length_resourcesi = strlen(this->resources[i]);
      memcpy(outbuffer + offset, &length_resourcesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->resources[i], length_resourcesi);
      offset += length_resourcesi;
      }
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
      uint32_t length_state;
      memcpy(&length_state, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_state; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_state-1]=0;
      this->state = (char *)(inbuffer + offset-1);
      offset += length_state;
      uint32_t length_type;
      memcpy(&length_type, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_type; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_type-1]=0;
      this->type = (char *)(inbuffer + offset-1);
      offset += length_type;
      uint32_t length_hardware_interface;
      memcpy(&length_hardware_interface, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_hardware_interface; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_hardware_interface-1]=0;
      this->hardware_interface = (char *)(inbuffer + offset-1);
      offset += length_hardware_interface;
      uint8_t resources_lengthT = *(inbuffer + offset++);
      if(resources_lengthT > resources_length)
        this->resources = (char**)realloc(this->resources, resources_lengthT * sizeof(char*));
      offset += 3;
      resources_length = resources_lengthT;
      for( uint8_t i = 0; i < resources_length; i++){
      uint32_t length_st_resources;
      memcpy(&length_st_resources, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_resources; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_resources-1]=0;
      this->st_resources = (char *)(inbuffer + offset-1);
      offset += length_st_resources;
        memcpy( &(this->resources[i]), &(this->st_resources), sizeof(char*));
      }
     return offset;
    }

    const char * getType(){ return "controller_manager_msgs/ControllerState"; };
    const char * getMD5(){ return "cac963cc68f4f5836765c108de0fc446"; };

  };

}
#endif