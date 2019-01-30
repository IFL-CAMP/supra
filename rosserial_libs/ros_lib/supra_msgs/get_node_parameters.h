#ifndef _ROS_SERVICE_get_node_parameters_h
#define _ROS_SERVICE_get_node_parameters_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "supra_msgs/parameter.h"

namespace supra_msgs
{

static const char GET_NODE_PARAMETERS[] = "supra_msgs/get_node_parameters";

  class get_node_parametersRequest : public ros::Msg
  {
    public:
      const char* nodeId;

    get_node_parametersRequest():
      nodeId("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_nodeId = strlen(this->nodeId);
      memcpy(outbuffer + offset, &length_nodeId, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->nodeId, length_nodeId);
      offset += length_nodeId;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_nodeId;
      memcpy(&length_nodeId, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_nodeId; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_nodeId-1]=0;
      this->nodeId = (char *)(inbuffer + offset-1);
      offset += length_nodeId;
     return offset;
    }

    const char * getType(){ return GET_NODE_PARAMETERS; };
    const char * getMD5(){ return "2bde23ce36b83ecf17071ca10832dd29"; };

  };

  class get_node_parametersResponse : public ros::Msg
  {
    public:
      uint8_t parameters_length;
      supra_msgs::parameter st_parameters;
      supra_msgs::parameter * parameters;

    get_node_parametersResponse():
      parameters_length(0), parameters(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = parameters_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < parameters_length; i++){
      offset += this->parameters[i].serialize(outbuffer + offset);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t parameters_lengthT = *(inbuffer + offset++);
      if(parameters_lengthT > parameters_length)
        this->parameters = (supra_msgs::parameter*)realloc(this->parameters, parameters_lengthT * sizeof(supra_msgs::parameter));
      offset += 3;
      parameters_length = parameters_lengthT;
      for( uint8_t i = 0; i < parameters_length; i++){
      offset += this->st_parameters.deserialize(inbuffer + offset);
        memcpy( &(this->parameters[i]), &(this->st_parameters), sizeof(supra_msgs::parameter));
      }
     return offset;
    }

    const char * getType(){ return GET_NODE_PARAMETERS; };
    const char * getMD5(){ return "58b9e0941e5908f7babcbe28a0d30398"; };

  };

  class get_node_parameters {
    public:
    typedef get_node_parametersRequest Request;
    typedef get_node_parametersResponse Response;
  };

}
#endif
