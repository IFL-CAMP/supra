#ifndef _ROS_SERVICE_GetModelState_h
#define _ROS_SERVICE_GetModelState_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Twist.h"

namespace gazebo_msgs
{

static const char GETMODELSTATE[] = "gazebo_msgs/GetModelState";

  class GetModelStateRequest : public ros::Msg
  {
    public:
      const char* model_name;
      const char* relative_entity_name;

    GetModelStateRequest():
      model_name(""),
      relative_entity_name("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_model_name = strlen(this->model_name);
      memcpy(outbuffer + offset, &length_model_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->model_name, length_model_name);
      offset += length_model_name;
      uint32_t length_relative_entity_name = strlen(this->relative_entity_name);
      memcpy(outbuffer + offset, &length_relative_entity_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->relative_entity_name, length_relative_entity_name);
      offset += length_relative_entity_name;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_model_name;
      memcpy(&length_model_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_model_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_model_name-1]=0;
      this->model_name = (char *)(inbuffer + offset-1);
      offset += length_model_name;
      uint32_t length_relative_entity_name;
      memcpy(&length_relative_entity_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_relative_entity_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_relative_entity_name-1]=0;
      this->relative_entity_name = (char *)(inbuffer + offset-1);
      offset += length_relative_entity_name;
     return offset;
    }

    const char * getType(){ return GETMODELSTATE; };
    const char * getMD5(){ return "19d412713cefe4a67437e17a951e759e"; };

  };

  class GetModelStateResponse : public ros::Msg
  {
    public:
      geometry_msgs::Pose pose;
      geometry_msgs::Twist twist;
      bool success;
      const char* status_message;

    GetModelStateResponse():
      pose(),
      twist(),
      success(0),
      status_message("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->pose.serialize(outbuffer + offset);
      offset += this->twist.serialize(outbuffer + offset);
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.real = this->success;
      *(outbuffer + offset + 0) = (u_success.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->success);
      uint32_t length_status_message = strlen(this->status_message);
      memcpy(outbuffer + offset, &length_status_message, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->status_message, length_status_message);
      offset += length_status_message;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->pose.deserialize(inbuffer + offset);
      offset += this->twist.deserialize(inbuffer + offset);
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.base = 0;
      u_success.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->success = u_success.real;
      offset += sizeof(this->success);
      uint32_t length_status_message;
      memcpy(&length_status_message, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_status_message; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_status_message-1]=0;
      this->status_message = (char *)(inbuffer + offset-1);
      offset += length_status_message;
     return offset;
    }

    const char * getType(){ return GETMODELSTATE; };
    const char * getMD5(){ return "1f8f991dc94e0cb27fe61383e0f576bb"; };

  };

  class GetModelState {
    public:
    typedef GetModelStateRequest Request;
    typedef GetModelStateResponse Response;
  };

}
#endif
