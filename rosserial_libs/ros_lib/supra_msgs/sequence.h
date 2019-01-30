#ifndef _ROS_SERVICE_sequence_h
#define _ROS_SERVICE_sequence_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace supra_msgs
{

static const char SEQUENCE[] = "supra_msgs/sequence";

  class sequenceRequest : public ros::Msg
  {
    public:
      bool sequenceActive;

    sequenceRequest():
      sequenceActive(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_sequenceActive;
      u_sequenceActive.real = this->sequenceActive;
      *(outbuffer + offset + 0) = (u_sequenceActive.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->sequenceActive);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_sequenceActive;
      u_sequenceActive.base = 0;
      u_sequenceActive.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->sequenceActive = u_sequenceActive.real;
      offset += sizeof(this->sequenceActive);
     return offset;
    }

    const char * getType(){ return SEQUENCE; };
    const char * getMD5(){ return "ddad3ed71117be7431f196de74ee955b"; };

  };

  class sequenceResponse : public ros::Msg
  {
    public:
      bool success;

    sequenceResponse():
      success(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.real = this->success;
      *(outbuffer + offset + 0) = (u_success.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->success);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.base = 0;
      u_success.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->success = u_success.real;
      offset += sizeof(this->success);
     return offset;
    }

    const char * getType(){ return SEQUENCE; };
    const char * getMD5(){ return "358e233cde0c8a8bcfea4ce193f8fc15"; };

  };

  class sequence {
    public:
    typedef sequenceRequest Request;
    typedef sequenceResponse Response;
  };

}
#endif
