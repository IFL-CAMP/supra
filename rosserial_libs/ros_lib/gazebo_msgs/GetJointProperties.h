#ifndef _ROS_SERVICE_GetJointProperties_h
#define _ROS_SERVICE_GetJointProperties_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace gazebo_msgs
{

static const char GETJOINTPROPERTIES[] = "gazebo_msgs/GetJointProperties";

  class GetJointPropertiesRequest : public ros::Msg
  {
    public:
      const char* joint_name;

    GetJointPropertiesRequest():
      joint_name("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_joint_name = strlen(this->joint_name);
      memcpy(outbuffer + offset, &length_joint_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->joint_name, length_joint_name);
      offset += length_joint_name;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_joint_name;
      memcpy(&length_joint_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_joint_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_joint_name-1]=0;
      this->joint_name = (char *)(inbuffer + offset-1);
      offset += length_joint_name;
     return offset;
    }

    const char * getType(){ return GETJOINTPROPERTIES; };
    const char * getMD5(){ return "0be1351618e1dc030eb7959d9a4902de"; };

  };

  class GetJointPropertiesResponse : public ros::Msg
  {
    public:
      uint8_t type;
      uint8_t damping_length;
      double st_damping;
      double * damping;
      uint8_t position_length;
      double st_position;
      double * position;
      uint8_t rate_length;
      double st_rate;
      double * rate;
      bool success;
      const char* status_message;
      enum { REVOLUTE =  0                 };
      enum { CONTINUOUS =  1                 };
      enum { PRISMATIC =  2                 };
      enum { FIXED =  3                 };
      enum { BALL =  4                 };
      enum { UNIVERSAL =  5                 };

    GetJointPropertiesResponse():
      type(0),
      damping_length(0), damping(NULL),
      position_length(0), position(NULL),
      rate_length(0), rate(NULL),
      success(0),
      status_message("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset + 0) = (this->type >> (8 * 0)) & 0xFF;
      offset += sizeof(this->type);
      *(outbuffer + offset++) = damping_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < damping_length; i++){
      union {
        double real;
        uint64_t base;
      } u_dampingi;
      u_dampingi.real = this->damping[i];
      *(outbuffer + offset + 0) = (u_dampingi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_dampingi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_dampingi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_dampingi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_dampingi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_dampingi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_dampingi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_dampingi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->damping[i]);
      }
      *(outbuffer + offset++) = position_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < position_length; i++){
      union {
        double real;
        uint64_t base;
      } u_positioni;
      u_positioni.real = this->position[i];
      *(outbuffer + offset + 0) = (u_positioni.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_positioni.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_positioni.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_positioni.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_positioni.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_positioni.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_positioni.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_positioni.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->position[i]);
      }
      *(outbuffer + offset++) = rate_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < rate_length; i++){
      union {
        double real;
        uint64_t base;
      } u_ratei;
      u_ratei.real = this->rate[i];
      *(outbuffer + offset + 0) = (u_ratei.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_ratei.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_ratei.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_ratei.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_ratei.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_ratei.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_ratei.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_ratei.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->rate[i]);
      }
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
      this->type =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->type);
      uint8_t damping_lengthT = *(inbuffer + offset++);
      if(damping_lengthT > damping_length)
        this->damping = (double*)realloc(this->damping, damping_lengthT * sizeof(double));
      offset += 3;
      damping_length = damping_lengthT;
      for( uint8_t i = 0; i < damping_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_damping;
      u_st_damping.base = 0;
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_damping = u_st_damping.real;
      offset += sizeof(this->st_damping);
        memcpy( &(this->damping[i]), &(this->st_damping), sizeof(double));
      }
      uint8_t position_lengthT = *(inbuffer + offset++);
      if(position_lengthT > position_length)
        this->position = (double*)realloc(this->position, position_lengthT * sizeof(double));
      offset += 3;
      position_length = position_lengthT;
      for( uint8_t i = 0; i < position_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_position;
      u_st_position.base = 0;
      u_st_position.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_position.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_position.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_position.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_position.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_position.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_position.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_position.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_position = u_st_position.real;
      offset += sizeof(this->st_position);
        memcpy( &(this->position[i]), &(this->st_position), sizeof(double));
      }
      uint8_t rate_lengthT = *(inbuffer + offset++);
      if(rate_lengthT > rate_length)
        this->rate = (double*)realloc(this->rate, rate_lengthT * sizeof(double));
      offset += 3;
      rate_length = rate_lengthT;
      for( uint8_t i = 0; i < rate_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_rate;
      u_st_rate.base = 0;
      u_st_rate.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_rate.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_rate.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_rate.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_rate.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_rate.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_rate.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_rate.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_rate = u_st_rate.real;
      offset += sizeof(this->st_rate);
        memcpy( &(this->rate[i]), &(this->st_rate), sizeof(double));
      }
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

    const char * getType(){ return GETJOINTPROPERTIES; };
    const char * getMD5(){ return "cd7b30a39faa372283dc94c5f6457f82"; };

  };

  class GetJointProperties {
    public:
    typedef GetJointPropertiesRequest Request;
    typedef GetJointPropertiesResponse Response;
  };

}
#endif
