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
      bool freeze;

    freeze():
      freeze(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_freeze;
      u_freeze.real = this->freeze;
      *(outbuffer + offset + 0) = (u_freeze.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->freeze);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_freeze;
      u_freeze.base = 0;
      u_freeze.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->freeze = u_freeze.real;
      offset += sizeof(this->freeze);
     return offset;
    }

    const char * getType(){ return "supra_msgs/freeze"; };
    const char * getMD5(){ return "4b161df344835f6e0ad165e599379cd6"; };

  };

}
#endif