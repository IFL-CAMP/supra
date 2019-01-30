#ifndef _ROS_SERVICE_NodeletList_h
#define _ROS_SERVICE_NodeletList_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace nodelet
{

static const char NODELETLIST[] = "nodelet/NodeletList";

  class NodeletListRequest : public ros::Msg
  {
    public:

    NodeletListRequest()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
     return offset;
    }

    const char * getType(){ return NODELETLIST; };
    const char * getMD5(){ return "d41d8cd98f00b204e9800998ecf8427e"; };

  };

  class NodeletListResponse : public ros::Msg
  {
    public:
      uint8_t nodelets_length;
      char* st_nodelets;
      char* * nodelets;

    NodeletListResponse():
      nodelets_length(0), nodelets(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = nodelets_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < nodelets_length; i++){
      uint32_t length_nodeletsi = strlen(this->nodelets[i]);
      memcpy(outbuffer + offset, &length_nodeletsi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->nodelets[i], length_nodeletsi);
      offset += length_nodeletsi;
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t nodelets_lengthT = *(inbuffer + offset++);
      if(nodelets_lengthT > nodelets_length)
        this->nodelets = (char**)realloc(this->nodelets, nodelets_lengthT * sizeof(char*));
      offset += 3;
      nodelets_length = nodelets_lengthT;
      for( uint8_t i = 0; i < nodelets_length; i++){
      uint32_t length_st_nodelets;
      memcpy(&length_st_nodelets, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_nodelets; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_nodelets-1]=0;
      this->st_nodelets = (char *)(inbuffer + offset-1);
      offset += length_st_nodelets;
        memcpy( &(this->nodelets[i]), &(this->st_nodelets), sizeof(char*));
      }
     return offset;
    }

    const char * getType(){ return NODELETLIST; };
    const char * getMD5(){ return "99c7b10e794f5600b8030e697e946ca7"; };

  };

  class NodeletList {
    public:
    typedef NodeletListRequest Request;
    typedef NodeletListResponse Response;
  };

}
#endif
