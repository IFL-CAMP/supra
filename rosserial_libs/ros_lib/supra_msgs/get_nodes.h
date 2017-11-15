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
      typedef int16_t _type_type;
      _type_type type;
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
      uint32_t ids_length;
      typedef char* _ids_type;
      _ids_type st_ids;
      _ids_type * ids;

    get_nodesResponse():
      ids_length(0), ids(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset + 0) = (this->ids_length >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->ids_length >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->ids_length >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->ids_length >> (8 * 3)) & 0xFF;
      offset += sizeof(this->ids_length);
      for( uint32_t i = 0; i < ids_length; i++){
      uint32_t length_idsi = strlen(this->ids[i]);
      varToArr(outbuffer + offset, length_idsi);
      offset += 4;
      memcpy(outbuffer + offset, this->ids[i], length_idsi);
      offset += length_idsi;
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t ids_lengthT = ((uint32_t) (*(inbuffer + offset))); 
      ids_lengthT |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1); 
      ids_lengthT |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2); 
      ids_lengthT |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3); 
      offset += sizeof(this->ids_length);
      if(ids_lengthT > ids_length)
        this->ids = (char**)realloc(this->ids, ids_lengthT * sizeof(char*));
      ids_length = ids_lengthT;
      for( uint32_t i = 0; i < ids_length; i++){
      uint32_t length_st_ids;
      arrToVar(length_st_ids, (inbuffer + offset));
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
