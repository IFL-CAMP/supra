#ifndef _ROS_supra_msgs_freeze_h
#define _ROS_supra_msgs_freeze_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace supra_msgs
{

  class freeze : public ros::Msg
  {
    public:
      bool freezeActive;

    freeze():
      freezeActive(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_freezeActive;
      u_freezeActive.real = this->freezeActive;
      *(outbuffer + offset + 0) = (u_freezeActive.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->freezeActive);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_freezeActive;
      u_freezeActive.base = 0;
      u_freezeActive.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->freezeActive = u_freezeActive.real;
      offset += sizeof(this->freezeActive);
     return offset;
    }

    const char * getType(){ return "supra_msgs/freeze"; };
    const char * getMD5(){ return "19c19546444bfbd4049fb289740c4ae3"; };

  };

}
#endif