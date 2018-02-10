#ifndef _ROS_SERVICE_AddDiagnostics_h
#define _ROS_SERVICE_AddDiagnostics_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace diagnostic_msgs
{

static const char ADDDIAGNOSTICS[] = "diagnostic_msgs/AddDiagnostics";

  class AddDiagnosticsRequest : public ros::Msg
  {
    public:
      const char* load_namespace;

    AddDiagnosticsRequest():
      load_namespace("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_load_namespace = strlen(this->load_namespace);
      memcpy(outbuffer + offset, &length_load_namespace, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->load_namespace, length_load_namespace);
      offset += length_load_namespace;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_load_namespace;
      memcpy(&length_load_namespace, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_load_namespace; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_load_namespace-1]=0;
      this->load_namespace = (char *)(inbuffer + offset-1);
      offset += length_load_namespace;
     return offset;
    }

    const char * getType(){ return ADDDIAGNOSTICS; };
    const char * getMD5(){ return "c26cf6e164288fbc6050d74f838bcdf0"; };

  };

  class AddDiagnosticsResponse : public ros::Msg
  {
    public:
      bool success;
      const char* message;

    AddDiagnosticsResponse():
      success(0),
      message("")
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
      uint32_t length_message = strlen(this->message);
      memcpy(outbuffer + offset, &length_message, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->message, length_message);
      offset += length_message;
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
      uint32_t length_message;
      memcpy(&length_message, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_message; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_message-1]=0;
      this->message = (char *)(inbuffer + offset-1);
      offset += length_message;
     return offset;
    }

    const char * getType(){ return ADDDIAGNOSTICS; };
    const char * getMD5(){ return "937c9679a518e3a18d831e57125ea522"; };

  };

  class AddDiagnostics {
    public:
    typedef AddDiagnosticsRequest Request;
    typedef AddDiagnosticsResponse Response;
  };

}
#endif
