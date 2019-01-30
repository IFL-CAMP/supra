#ifndef _ROS_SERVICE_get_nodes_h
#define _ROS_SERVICE_get_nodes_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace supra_msgs
{

static const char GET_NODES[] = "supra_msgs/get_nodes";

  class get_nodesRequest : public ros::Msg
  {
    public:
      int16_t type;
      enum { typeInput = 0 };
      enum { typeOutput = 1 };
      enum { typeAll = 2 };

    get_nodesRequest():
      type(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        int16_t real;
        uint16_t base;
      } u_type;
      u_type.real = this->type;
      *(outbuffer + offset + 0) = (u_type.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_type.base >> (8 * 1)) & 0xFF;
      offset += sizeof(this->type);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        int16_t real;
        uint16_t base;
      } u_type;
      u_type.base = 0;
      u_type.base |= ((uint16_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_type.base |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->type = u_type.real;
      offset += sizeof(this->type);
     return offset;
    }

    const char * getType(){ return GET_NODES; };
    const char * getMD5(){ return "d2af7cbe3a0dfb9222c784295af95b1d"; };

  };

  class get_nodesResponse : public ros::Msg
  {
    public:
      uint8_t ids_length;
      char* st_ids;
      char* * ids;

    get_nodesResponse():
      ids_length(0), ids(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = ids_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < ids_length; i++){
      uint32_t length_idsi = strlen(this->ids[i]);
      memcpy(outbuffer + offset, &length_idsi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->ids[i], length_idsi);
      offset += length_idsi;
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t ids_lengthT = *(inbuffer + offset++);
      if(ids_lengthT > ids_length)
        this->ids = (char**)realloc(this->ids, ids_lengthT * sizeof(char*));
      offset += 3;
      ids_length = ids_lengthT;
      for( uint8_t i = 0; i < ids_length; i++){
      uint32_t length_st_ids;
      memcpy(&length_st_ids, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_ids; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_ids-1]=0;
      this->st_ids = (char *)(inbuffer + offset-1);
      offset += length_st_ids;
        memcpy( &(this->ids[i]), &(this->st_ids), sizeof(char*));
      }
     return offset;
    }

    const char * getType(){ return GET_NODES; };
    const char * getMD5(){ return "31314a1125a2ca69ddc92cdc117c989c"; };

  };

  class get_nodes {
    public:
    typedef get_nodesRequest Request;
    typedef get_nodesResponse Response;
  };

}
#endif
