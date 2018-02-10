#ifndef _ROS_stereo_msgs_DisparityImage_h
#define _ROS_stereo_msgs_DisparityImage_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/RegionOfInterest.h"

namespace stereo_msgs
{

  class DisparityImage : public ros::Msg
  {
    public:
      std_msgs::Header header;
      sensor_msgs::Image image;
      float f;
      float T;
      sensor_msgs::RegionOfInterest valid_window;
      float min_disparity;
      float max_disparity;
      float delta_d;

    DisparityImage():
      header(),
      image(),
      f(0),
      T(0),
      valid_window(),
      min_disparity(0),
      max_disparity(0),
      delta_d(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      offset += this->image.serialize(outbuffer + offset);
      union {
        float real;
        uint32_t base;
      } u_f;
      u_f.real = this->f;
      *(outbuffer + offset + 0) = (u_f.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_f.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_f.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_f.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->f);
      union {
        float real;
        uint32_t base;
      } u_T;
      u_T.real = this->T;
      *(outbuffer + offset + 0) = (u_T.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_T.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_T.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_T.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->T);
      offset += this->valid_window.serialize(outbuffer + offset);
      union {
        float real;
        uint32_t base;
      } u_min_disparity;
      u_min_disparity.real = this->min_disparity;
      *(outbuffer + offset + 0) = (u_min_disparity.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_min_disparity.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_min_disparity.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_min_disparity.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->min_disparity);
      union {
        float real;
        uint32_t base;
      } u_max_disparity;
      u_max_disparity.real = this->max_disparity;
      *(outbuffer + offset + 0) = (u_max_disparity.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_max_disparity.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_max_disparity.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_max_disparity.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->max_disparity);
      union {
        float real;
        uint32_t base;
      } u_delta_d;
      u_delta_d.real = this->delta_d;
      *(outbuffer + offset + 0) = (u_delta_d.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_delta_d.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_delta_d.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_delta_d.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->delta_d);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      offset += this->image.deserialize(inbuffer + offset);
      union {
        float real;
        uint32_t base;
      } u_f;
      u_f.base = 0;
      u_f.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_f.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_f.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_f.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->f = u_f.real;
      offset += sizeof(this->f);
      union {
        float real;
        uint32_t base;
      } u_T;
      u_T.base = 0;
      u_T.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_T.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_T.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_T.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->T = u_T.real;
      offset += sizeof(this->T);
      offset += this->valid_window.deserialize(inbuffer + offset);
      union {
        float real;
        uint32_t base;
      } u_min_disparity;
      u_min_disparity.base = 0;
      u_min_disparity.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_min_disparity.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_min_disparity.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_min_disparity.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->min_disparity = u_min_disparity.real;
      offset += sizeof(this->min_disparity);
      union {
        float real;
        uint32_t base;
      } u_max_disparity;
      u_max_disparity.base = 0;
      u_max_disparity.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_max_disparity.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_max_disparity.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_max_disparity.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->max_disparity = u_max_disparity.real;
      offset += sizeof(this->max_disparity);
      union {
        float real;
        uint32_t base;
      } u_delta_d;
      u_delta_d.base = 0;
      u_delta_d.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_delta_d.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_delta_d.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_delta_d.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->delta_d = u_delta_d.real;
      offset += sizeof(this->delta_d);
     return offset;
    }

    const char * getType(){ return "stereo_msgs/DisparityImage"; };
    const char * getMD5(){ return "04a177815f75271039fa21f16acad8c9"; };

  };

}
#endif