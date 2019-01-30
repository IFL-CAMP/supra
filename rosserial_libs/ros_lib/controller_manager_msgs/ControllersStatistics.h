#ifndef _ROS_controller_manager_msgs_ControllersStatistics_h
#define _ROS_controller_manager_msgs_ControllersStatistics_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"
#include "controller_manager_msgs/ControllerStatistics.h"

namespace controller_manager_msgs
{

  class ControllersStatistics : public ros::Msg
  {
    public:
      std_msgs::Header header;
      uint8_t controller_length;
      controller_manager_msgs::ControllerStatistics st_controller;
      controller_manager_msgs::ControllerStatistics * controller;

    ControllersStatistics():
      header(),
      controller_length(0), controller(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
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
      offset += this->header.deserialize(inbuffer + offset);
      uint8_t controller_lengthT = *(inbuffer + offset++);
      if(controller_lengthT > controller_length)
        this->controller = (controller_manager_msgs::ControllerStatistics*)realloc(this->controller, controller_lengthT * sizeof(controller_manager_msgs::ControllerStatistics));
      offset += 3;
      controller_length = controller_lengthT;
      for( uint8_t i = 0; i < controller_length; i++){
      offset += this->st_controller.deserialize(inbuffer + offset);
        memcpy( &(this->controller[i]), &(this->st_controller), sizeof(controller_manager_msgs::ControllerStatistics));
      }
     return offset;
    }

    const char * getType(){ return "controller_manager_msgs/ControllersStatistics"; };
    const char * getMD5(){ return "a154c347736773e3700d1719105df29d"; };

  };

}
#endif