#ifndef _ROS_visualization_msgs_InteractiveMarker_h
#define _ROS_visualization_msgs_InteractiveMarker_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"
#include "geometry_msgs/Pose.h"
#include "visualization_msgs/MenuEntry.h"
#include "visualization_msgs/InteractiveMarkerControl.h"

namespace visualization_msgs
{

  class InteractiveMarker : public ros::Msg
  {
    public:
      std_msgs::Header header;
      geometry_msgs::Pose pose;
      const char* name;
      const char* description;
      float scale;
      uint8_t menu_entries_length;
      visualization_msgs::MenuEntry st_menu_entries;
      visualization_msgs::MenuEntry * menu_entries;
      uint8_t controls_length;
      visualization_msgs::InteractiveMarkerControl st_controls;
      visualization_msgs::InteractiveMarkerControl * controls;

    InteractiveMarker():
      header(),
      pose(),
      name(""),
      description(""),
      scale(0),
      menu_entries_length(0), menu_entries(NULL),
      controls_length(0), controls(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      offset += this->pose.serialize(outbuffer + offset);
      uint32_t length_name = strlen(this->name);
      memcpy(outbuffer + offset, &length_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->name, length_name);
      offset += length_name;
      uint32_t length_description = strlen(this->description);
      memcpy(outbuffer + offset, &length_description, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->description, length_description);
      offset += length_description;
      union {
        float real;
        uint32_t base;
      } u_scale;
      u_scale.real = this->scale;
      *(outbuffer + offset + 0) = (u_scale.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_scale.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_scale.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_scale.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->scale);
      *(outbuffer + offset++) = menu_entries_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < menu_entries_length; i++){
      offset += this->menu_entries[i].serialize(outbuffer + offset);
      }
      *(outbuffer + offset++) = controls_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < controls_length; i++){
      offset += this->controls[i].serialize(outbuffer + offset);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      offset += this->pose.deserialize(inbuffer + offset);
      uint32_t length_name;
      memcpy(&length_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_name-1]=0;
      this->name = (char *)(inbuffer + offset-1);
      offset += length_name;
      uint32_t length_description;
      memcpy(&length_description, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_description; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_description-1]=0;
      this->description = (char *)(inbuffer + offset-1);
      offset += length_description;
      union {
        float real;
        uint32_t base;
      } u_scale;
      u_scale.base = 0;
      u_scale.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_scale.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_scale.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_scale.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->scale = u_scale.real;
      offset += sizeof(this->scale);
      uint8_t menu_entries_lengthT = *(inbuffer + offset++);
      if(menu_entries_lengthT > menu_entries_length)
        this->menu_entries = (visualization_msgs::MenuEntry*)realloc(this->menu_entries, menu_entries_lengthT * sizeof(visualization_msgs::MenuEntry));
      offset += 3;
      menu_entries_length = menu_entries_lengthT;
      for( uint8_t i = 0; i < menu_entries_length; i++){
      offset += this->st_menu_entries.deserialize(inbuffer + offset);
        memcpy( &(this->menu_entries[i]), &(this->st_menu_entries), sizeof(visualization_msgs::MenuEntry));
      }
      uint8_t controls_lengthT = *(inbuffer + offset++);
      if(controls_lengthT > controls_length)
        this->controls = (visualization_msgs::InteractiveMarkerControl*)realloc(this->controls, controls_lengthT * sizeof(visualization_msgs::InteractiveMarkerControl));
      offset += 3;
      controls_length = controls_lengthT;
      for( uint8_t i = 0; i < controls_length; i++){
      offset += this->st_controls.deserialize(inbuffer + offset);
        memcpy( &(this->controls[i]), &(this->st_controls), sizeof(visualization_msgs::InteractiveMarkerControl));
      }
     return offset;
    }

    const char * getType(){ return "visualization_msgs/InteractiveMarker"; };
    const char * getMD5(){ return "311bd5f6cd6a20820ac0ba84315f4e22"; };

  };

}
#endif