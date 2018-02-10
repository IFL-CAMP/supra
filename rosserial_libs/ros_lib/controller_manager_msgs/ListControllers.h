#ifndef _ROS_SERVICE_ListControllers_h
#define _ROS_SERVICE_ListControllers_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "controller_manager_msgs/ControllerState.h"

namespace controller_manager_msgs
{

static const char LISTCONTROLLERS[] = "controller_manager_msgs/ListControllers";

  class ListControllersRequest : public ros::Msg
  {
    public:

    ListControllersRequest()
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

    const char * getType(){ return LISTCONTROLLERS; };
    const char * getMD5(){ return "d41d8cd98f00b204e9800998ecf8427e"; };

  };

  class ListControllersResponse : public ros::Msg
  {
    public:
      uint8_t controller_length;
      controller_manager_msgs::ControllerState st_controller;
      controller_manager_msgs::ControllerState * controller;

    ListControllersResponse():
      controller_length(0), controller(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = controller_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < controller_length; i++){
      offset += this->controller[i].serialize(outbuffer + offset);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t controller_lengthT = *(inbuffer + offset++);
      if(controller_lengthT > controller_length)
        this->controller = (controller_manager_msgs::ControllerState*)realloc(this->controller, controller_lengthT * sizeof(controller_manager_msgs::ControllerState));
      offset += 3;
      controller_length = controller_lengthT;
      for( uint8_t i = 0; i < controller_length; i++){
      offset += this->st_controller.deserialize(inbuffer + offset);
        memcpy( &(this->controller[i]), &(this->st_controller), sizeof(controller_manager_msgs::ControllerState));
      }
     return offset;
    }

    const char * getType(){ return LISTCONTROLLERS; };
    const char * getMD5(){ return "12c85fca1984c8ec86264f3d00b938f2"; };

  };

  class ListControllers {
    public:
    typedef ListControllersRequest Request;
    typedef ListControllersResponse Response;
  };

}
#endif
