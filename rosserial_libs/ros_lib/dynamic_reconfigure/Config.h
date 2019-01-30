#ifndef _ROS_dynamic_reconfigure_Config_h
#define _ROS_dynamic_reconfigure_Config_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "dynamic_reconfigure/BoolParameter.h"
#include "dynamic_reconfigure/IntParameter.h"
#include "dynamic_reconfigure/StrParameter.h"
#include "dynamic_reconfigure/DoubleParameter.h"
#include "dynamic_reconfigure/GroupState.h"

namespace dynamic_reconfigure
{

  class Config : public ros::Msg
  {
    public:
      uint8_t bools_length;
      dynamic_reconfigure::BoolParameter st_bools;
      dynamic_reconfigure::BoolParameter * bools;
      uint8_t ints_length;
      dynamic_reconfigure::IntParameter st_ints;
      dynamic_reconfigure::IntParameter * ints;
      uint8_t strs_length;
      dynamic_reconfigure::StrParameter st_strs;
      dynamic_reconfigure::StrParameter * strs;
      uint8_t doubles_length;
      dynamic_reconfigure::DoubleParameter st_doubles;
      dynamic_reconfigure::DoubleParameter * doubles;
      uint8_t groups_length;
      dynamic_reconfigure::GroupState st_groups;
      dynamic_reconfigure::GroupState * groups;

    Config():
      bools_length(0), bools(NULL),
      ints_length(0), ints(NULL),
      strs_length(0), strs(NULL),
      doubles_length(0), doubles(NULL),
      groups_length(0), groups(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = bools_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < bools_length; i++){
      offset += this->bools[i].serialize(outbuffer + offset);
      }
      *(outbuffer + offset++) = ints_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < ints_length; i++){
      offset += this->ints[i].serialize(outbuffer + offset);
      }
      *(outbuffer + offset++) = strs_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < strs_length; i++){
      offset += this->strs[i].serialize(outbuffer + offset);
      }
      *(outbuffer + offset++) = doubles_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < doubles_length; i++){
      offset += this->doubles[i].serialize(outbuffer + offset);
      }
      *(outbuffer + offset++) = groups_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < groups_length; i++){
      offset += this->groups[i].serialize(outbuffer + offset);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t bools_lengthT = *(inbuffer + offset++);
      if(bools_lengthT > bools_length)
        this->bools = (dynamic_reconfigure::BoolParameter*)realloc(this->bools, bools_lengthT * sizeof(dynamic_reconfigure::BoolParameter));
      offset += 3;
      bools_length = bools_lengthT;
      for( uint8_t i = 0; i < bools_length; i++){
      offset += this->st_bools.deserialize(inbuffer + offset);
        memcpy( &(this->bools[i]), &(this->st_bools), sizeof(dynamic_reconfigure::BoolParameter));
      }
      uint8_t ints_lengthT = *(inbuffer + offset++);
      if(ints_lengthT > ints_length)
        this->ints = (dynamic_reconfigure::IntParameter*)realloc(this->ints, ints_lengthT * sizeof(dynamic_reconfigure::IntParameter));
      offset += 3;
      ints_length = ints_lengthT;
      for( uint8_t i = 0; i < ints_length; i++){
      offset += this->st_ints.deserialize(inbuffer + offset);
        memcpy( &(this->ints[i]), &(this->st_ints), sizeof(dynamic_reconfigure::IntParameter));
      }
      uint8_t strs_lengthT = *(inbuffer + offset++);
      if(strs_lengthT > strs_length)
        this->strs = (dynamic_reconfigure::StrParameter*)realloc(this->strs, strs_lengthT * sizeof(dynamic_reconfigure::StrParameter));
      offset += 3;
      strs_length = strs_lengthT;
      for( uint8_t i = 0; i < strs_length; i++){
      offset += this->st_strs.deserialize(inbuffer + offset);
        memcpy( &(this->strs[i]), &(this->st_strs), sizeof(dynamic_reconfigure::StrParameter));
      }
      uint8_t doubles_lengthT = *(inbuffer + offset++);
      if(doubles_lengthT > doubles_length)
        this->doubles = (dynamic_reconfigure::DoubleParameter*)realloc(this->doubles, doubles_lengthT * sizeof(dynamic_reconfigure::DoubleParameter));
      offset += 3;
      doubles_length = doubles_lengthT;
      for( uint8_t i = 0; i < doubles_length; i++){
      offset += this->st_doubles.deserialize(inbuffer + offset);
        memcpy( &(this->doubles[i]), &(this->st_doubles), sizeof(dynamic_reconfigure::DoubleParameter));
      }
      uint8_t groups_lengthT = *(inbuffer + offset++);
      if(groups_lengthT > groups_length)
        this->groups = (dynamic_reconfigure::GroupState*)realloc(this->groups, groups_lengthT * sizeof(dynamic_reconfigure::GroupState));
      offset += 3;
      groups_length = groups_lengthT;
      for( uint8_t i = 0; i < groups_length; i++){
      offset += this->st_groups.deserialize(inbuffer + offset);
        memcpy( &(this->groups[i]), &(this->st_groups), sizeof(dynamic_reconfigure::GroupState));
      }
     return offset;
    }

    const char * getType(){ return "dynamic_reconfigure/Config"; };
    const char * getMD5(){ return "958f16a05573709014982821e6822580"; };

  };

}
#endif