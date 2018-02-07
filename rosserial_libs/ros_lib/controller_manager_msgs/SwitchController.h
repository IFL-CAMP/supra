#ifndef _ROS_SERVICE_SwitchController_h
#define _ROS_SERVICE_SwitchController_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace controller_manager_msgs
{

static const char SWITCHCONTROLLER[] = "controller_manager_msgs/SwitchController";

  class SwitchControllerRequest : public ros::Msg
  {
    public:
      uint8_t start_controllers_length;
      char* st_start_controllers;
      char* * start_controllers;
      uint8_t stop_controllers_length;
      char* st_stop_controllers;
      char* * stop_controllers;
      int32_t strictness;
      enum { BEST_EFFORT = 1 };
      enum { STRICT = 2 };

    SwitchControllerRequest():
      start_controllers_length(0), start_controllers(NULL),
      stop_controllers_length(0), stop_controllers(NULL),
      strictness(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = start_controllers_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < start_controllers_length; i++){
      uint32_t length_start_controllersi = strlen(this->start_controllers[i]);
      memcpy(outbuffer + offset, &length_start_controllersi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->start_controllers[i], length_start_controllersi);
      offset += length_start_controllersi;
      }
      *(outbuffer + offset++) = stop_controllers_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < stop_controllers_length; i++){
      uint32_t length_stop_controllersi = strlen(this->stop_controllers[i]);
      memcpy(outbuffer + offset, &length_stop_controllersi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->stop_controllers[i], length_stop_controllersi);
      offset += length_stop_controllersi;
      }
      union {
        int32_t real;
        uint32_t base;
      } u_strictness;
      u_strictness.real = this->strictness;
      *(outbuffer + offset + 0) = (u_strictness.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_strictness.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_strictness.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_strictness.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->strictness);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t start_controllers_lengthT = *(inbuffer + offset++);
      if(start_controllers_lengthT > start_controllers_length)
        this->start_controllers = (char**)realloc(this->start_controllers, start_controllers_lengthT * sizeof(char*));
      offset += 3;
      start_controllers_length = start_controllers_lengthT;
      for( uint8_t i = 0; i < start_controllers_length; i++){
      uint32_t length_st_start_controllers;
      memcpy(&length_st_start_controllers, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_start_controllers; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_start_controllers-1]=0;
      this->st_start_controllers = (char *)(inbuffer + offset-1);
      offset += length_st_start_controllers;
        memcpy( &(this->start_controllers[i]), &(this->st_start_controllers), sizeof(char*));
      }
      uint8_t stop_controllers_lengthT = *(inbuffer + offset++);
      if(stop_controllers_lengthT > stop_controllers_length)
        this->stop_controllers = (char**)realloc(this->stop_controllers, stop_controllers_lengthT * sizeof(char*));
      offset += 3;
      stop_controllers_length = stop_controllers_lengthT;
      for( uint8_t i = 0; i < stop_controllers_length; i++){
      uint32_t length_st_stop_controllers;
      memcpy(&length_st_stop_controllers, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_stop_controllers; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_stop_controllers-1]=0;
      this->st_stop_controllers = (char *)(inbuffer + offset-1);
      offset += length_st_stop_controllers;
        memcpy( &(this->stop_controllers[i]), &(this->st_stop_controllers), sizeof(char*));
      }
      union {
        int32_t real;
        uint32_t base;
      } u_strictness;
      u_strictness.base = 0;
      u_strictness.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_strictness.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_strictness.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_strictness.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->strictness = u_strictness.real;
      offset += sizeof(this->strictness);
     return offset;
    }

    const char * getType(){ return SWITCHCONTROLLER; };
    const char * getMD5(){ return "434da54adc434a5af5743ed711fd6ba1"; };

  };

  class SwitchControllerResponse : public ros::Msg
  {
    public:
      bool ok;

    SwitchControllerResponse():
      ok(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_ok;
      u_ok.real = this->ok;
      *(outbuffer + offset + 0) = (u_ok.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->ok);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_ok;
      u_ok.base = 0;
      u_ok.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->ok = u_ok.real;
      offset += sizeof(this->ok);
     return offset;
    }

    const char * getType(){ return SWITCHCONTROLLER; };
    const char * getMD5(){ return "6f6da3883749771fac40d6deb24a8c02"; };

  };

  class SwitchController {
    public:
    typedef SwitchControllerRequest Request;
    typedef SwitchControllerResponse Response;
  };

}
#endif
