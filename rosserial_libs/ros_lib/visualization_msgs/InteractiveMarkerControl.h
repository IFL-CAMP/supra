#ifndef _ROS_visualization_msgs_InteractiveMarkerControl_h
#define _ROS_visualization_msgs_InteractiveMarkerControl_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "geometry_msgs/Quaternion.h"
#include "visualization_msgs/Marker.h"

namespace visualization_msgs
{

  class InteractiveMarkerControl : public ros::Msg
  {
    public:
      const char* name;
      geometry_msgs::Quaternion orientation;
      uint8_t orientation_mode;
      uint8_t interaction_mode;
      bool always_visible;
      uint8_t markers_length;
      visualization_msgs::Marker st_markers;
      visualization_msgs::Marker * markers;
      bool independent_marker_orientation;
      const char* description;
      enum { INHERIT =  0 };
      enum { FIXED =  1 };
      enum { VIEW_FACING =  2 };
      enum { NONE =  0 };
      enum { MENU =  1 };
      enum { BUTTON =  2 };
      enum { MOVE_AXIS =  3 };
      enum { MOVE_PLANE =  4 };
      enum { ROTATE_AXIS =  5 };
      enum { MOVE_ROTATE =  6 };
      enum { MOVE_3D =  7 };
      enum { ROTATE_3D =  8 };
      enum { MOVE_ROTATE_3D =  9 };

    InteractiveMarkerControl():
      name(""),
      orientation(),
      orientation_mode(0),
      interaction_mode(0),
      always_visible(0),
      markers_length(0), markers(NULL),
      independent_marker_orientation(0),
      description("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_name = strlen(this->name);
      memcpy(outbuffer + offset, &length_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->name, length_name);
      offset += length_name;
      offset += this->orientation.serialize(outbuffer + offset);
      *(outbuffer + offset + 0) = (this->orientation_mode >> (8 * 0)) & 0xFF;
      offset += sizeof(this->orientation_mode);
      *(outbuffer + offset + 0) = (this->interaction_mode >> (8 * 0)) & 0xFF;
      offset += sizeof(this->interaction_mode);
      union {
        bool real;
        uint8_t base;
      } u_always_visible;
      u_always_visible.real = this->always_visible;
      *(outbuffer + offset + 0) = (u_always_visible.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->always_visible);
      *(outbuffer + offset++) = markers_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < markers_length; i++){
      offset += this->markers[i].serialize(outbuffer + offset);
      }
      union {
        bool real;
        uint8_t base;
      } u_independent_marker_orientation;
      u_independent_marker_orientation.real = this->independent_marker_orientation;
      *(outbuffer + offset + 0) = (u_independent_marker_orientation.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->independent_marker_orientation);
      uint32_t length_description = strlen(this->description);
      memcpy(outbuffer + offset, &length_description, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->description, length_description);
      offset += length_description;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_name;
      memcpy(&length_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_name-1]=0;
      this->name = (char *)(inbuffer + offset-1);
      offset += length_name;
      offset += this->orientation.deserialize(inbuffer + offset);
      this->orientation_mode =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->orientation_mode);
      this->interaction_mode =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->interaction_mode);
      union {
        bool real;
        uint8_t base;
      } u_always_visible;
      u_always_visible.base = 0;
      u_always_visible.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->always_visible = u_always_visible.real;
      offset += sizeof(this->always_visible);
      uint8_t markers_lengthT = *(inbuffer + offset++);
      if(markers_lengthT > markers_length)
        this->markers = (visualization_msgs::Marker*)realloc(this->markers, markers_lengthT * sizeof(visualization_msgs::Marker));
      offset += 3;
      markers_length = markers_lengthT;
      for( uint8_t i = 0; i < markers_length; i++){
      offset += this->st_markers.deserialize(inbuffer + offset);
        memcpy( &(this->markers[i]), &(this->st_markers), sizeof(visualization_msgs::Marker));
      }
      union {
        bool real;
        uint8_t base;
      } u_independent_marker_orientation;
      u_independent_marker_orientation.base = 0;
      u_independent_marker_orientation.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->independent_marker_orientation = u_independent_marker_orientation.real;
      offset += sizeof(this->independent_marker_orientation);
      uint32_t length_description;
      memcpy(&length_description, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_description; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_description-1]=0;
      this->description = (char *)(inbuffer + offset-1);
      offset += length_description;
     return offset;
    }

    const char * getType(){ return "visualization_msgs/InteractiveMarkerControl"; };
    const char * getMD5(){ return "e3a939c98cdd4f709d8e1dec2a9c3f37"; };

  };

}
#endif