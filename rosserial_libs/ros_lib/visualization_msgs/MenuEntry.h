#ifndef _ROS_visualization_msgs_MenuEntry_h
#define _ROS_visualization_msgs_MenuEntry_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace visualization_msgs
{

  class MenuEntry : public ros::Msg
  {
    public:
      uint32_t id;
      uint32_t parent_id;
      const char* title;
      const char* command;
      uint8_t command_type;
      enum { FEEDBACK = 0 };
      enum { ROSRUN = 1 };
      enum { ROSLAUNCH = 2 };

    MenuEntry():
      id(0),
      parent_id(0),
      title(""),
      command(""),
      command_type(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset + 0) = (this->id >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->id >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->id >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->id >> (8 * 3)) & 0xFF;
      offset += sizeof(this->id);
      *(outbuffer + offset + 0) = (this->parent_id >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->parent_id >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->parent_id >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->parent_id >> (8 * 3)) & 0xFF;
      offset += sizeof(this->parent_id);
      uint32_t length_title = strlen(this->title);
      memcpy(outbuffer + offset, &length_title, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->title, length_title);
      offset += length_title;
      uint32_t length_command = strlen(this->command);
      memcpy(outbuffer + offset, &length_command, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->command, length_command);
      offset += length_command;
      *(outbuffer + offset + 0) = (this->command_type >> (8 * 0)) & 0xFF;
      offset += sizeof(this->command_type);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      this->id =  ((uint32_t) (*(inbuffer + offset)));
      this->id |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->id |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->id |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->id);
      this->parent_id =  ((uint32_t) (*(inbuffer + offset)));
      this->parent_id |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->parent_id |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->parent_id |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->parent_id);
      uint32_t length_title;
      memcpy(&length_title, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_title; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_title-1]=0;
      this->title = (char *)(inbuffer + offset-1);
      offset += length_title;
      uint32_t length_command;
      memcpy(&length_command, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_command; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_command-1]=0;
      this->command = (char *)(inbuffer + offset-1);
      offset += length_command;
      this->command_type =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->command_type);
     return offset;
    }

    const char * getType(){ return "visualization_msgs/MenuEntry"; };
    const char * getMD5(){ return "b90ec63024573de83b57aa93eb39be2d"; };

  };

}
#endif