#ifndef _ROS_shape_msgs_Mesh_h
#define _ROS_shape_msgs_Mesh_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "shape_msgs/MeshTriangle.h"
#include "geometry_msgs/Point.h"

namespace shape_msgs
{

  class Mesh : public ros::Msg
  {
    public:
      uint8_t triangles_length;
      shape_msgs::MeshTriangle st_triangles;
      shape_msgs::MeshTriangle * triangles;
      uint8_t vertices_length;
      geometry_msgs::Point st_vertices;
      geometry_msgs::Point * vertices;

    Mesh():
      triangles_length(0), triangles(NULL),
      vertices_length(0), vertices(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = triangles_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < triangles_length; i++){
      offset += this->triangles[i].serialize(outbuffer + offset);
      }
      *(outbuffer + offset++) = vertices_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < vertices_length; i++){
      offset += this->vertices[i].serialize(outbuffer + offset);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t triangles_lengthT = *(inbuffer + offset++);
      if(triangles_lengthT > triangles_length)
        this->triangles = (shape_msgs::MeshTriangle*)realloc(this->triangles, triangles_lengthT * sizeof(shape_msgs::MeshTriangle));
      offset += 3;
      triangles_length = triangles_lengthT;
      for( uint8_t i = 0; i < triangles_length; i++){
      offset += this->st_triangles.deserialize(inbuffer + offset);
        memcpy( &(this->triangles[i]), &(this->st_triangles), sizeof(shape_msgs::MeshTriangle));
      }
      uint8_t vertices_lengthT = *(inbuffer + offset++);
      if(vertices_lengthT > vertices_length)
        this->vertices = (geometry_msgs::Point*)realloc(this->vertices, vertices_lengthT * sizeof(geometry_msgs::Point));
      offset += 3;
      vertices_length = vertices_lengthT;
      for( uint8_t i = 0; i < vertices_length; i++){
      offset += this->st_vertices.deserialize(inbuffer + offset);
        memcpy( &(this->vertices[i]), &(this->st_vertices), sizeof(geometry_msgs::Point));
      }
     return offset;
    }

    const char * getType(){ return "shape_msgs/Mesh"; };
    const char * getMD5(){ return "1ffdae9486cd3316a121c578b47a85cc"; };

  };

}
#endif