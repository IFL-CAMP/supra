#ifndef _ROS_supra_msgs_UInt8Image_h
#define _ROS_supra_msgs_UInt8Image_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"
#include "geometry_msgs/PoseStamped.h"

namespace supra_msgs
{

  class UInt8Image : public ros::Msg
  {
    public:
      std_msgs::Header header;
      const char* modality;
      uint32_t width;
      uint32_t height;
      uint32_t depth;
      geometry_msgs::PoseStamped origin;
      float resX;
      float resY;
      float resZ;
      uint8_t volume_length;
      uint8_t st_volume;
      uint8_t * volume;

    UInt8Image():
      header(),
      modality(""),
      width(0),
      height(0),
      depth(0),
      origin(),
      resX(0),
      resY(0),
      resZ(0),
      volume_length(0), volume(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      uint32_t length_modality = strlen(this->modality);
      memcpy(outbuffer + offset, &length_modality, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->modality, length_modality);
      offset += length_modality;
      *(outbuffer + offset + 0) = (this->width >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->width >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->width >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->width >> (8 * 3)) & 0xFF;
      offset += sizeof(this->width);
      *(outbuffer + offset + 0) = (this->height >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->height >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->height >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->height >> (8 * 3)) & 0xFF;
      offset += sizeof(this->height);
      *(outbuffer + offset + 0) = (this->depth >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->depth >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->depth >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->depth >> (8 * 3)) & 0xFF;
      offset += sizeof(this->depth);
      offset += this->origin.serialize(outbuffer + offset);
      union {
        float real;
        uint32_t base;
      } u_resX;
      u_resX.real = this->resX;
      *(outbuffer + offset + 0) = (u_resX.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_resX.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_resX.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_resX.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->resX);
      union {
        float real;
        uint32_t base;
      } u_resY;
      u_resY.real = this->resY;
      *(outbuffer + offset + 0) = (u_resY.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_resY.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_resY.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_resY.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->resY);
      union {
        float real;
        uint32_t base;
      } u_resZ;
      u_resZ.real = this->resZ;
      *(outbuffer + offset + 0) = (u_resZ.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_resZ.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_resZ.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_resZ.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->resZ);
      *(outbuffer + offset++) = volume_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < volume_length; i++){
      *(outbuffer + offset + 0) = (this->volume[i] >> (8 * 0)) & 0xFF;
      offset += sizeof(this->volume[i]);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      uint32_t length_modality;
      memcpy(&length_modality, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_modality; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_modality-1]=0;
      this->modality = (char *)(inbuffer + offset-1);
      offset += length_modality;
      this->width =  ((uint32_t) (*(inbuffer + offset)));
      this->width |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->width |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->width |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->width);
      this->height =  ((uint32_t) (*(inbuffer + offset)));
      this->height |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->height |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->height |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->height);
      this->depth =  ((uint32_t) (*(inbuffer + offset)));
      this->depth |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->depth |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->depth |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->depth);
      offset += this->origin.deserialize(inbuffer + offset);
      union {
        float real;
        uint32_t base;
      } u_resX;
      u_resX.base = 0;
      u_resX.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_resX.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_resX.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_resX.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->resX = u_resX.real;
      offset += sizeof(this->resX);
      union {
        float real;
        uint32_t base;
      } u_resY;
      u_resY.base = 0;
      u_resY.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_resY.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_resY.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_resY.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->resY = u_resY.real;
      offset += sizeof(this->resY);
      union {
        float real;
        uint32_t base;
      } u_resZ;
      u_resZ.base = 0;
      u_resZ.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_resZ.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_resZ.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_resZ.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->resZ = u_resZ.real;
      offset += sizeof(this->resZ);
      uint8_t volume_lengthT = *(inbuffer + offset++);
      if(volume_lengthT > volume_length)
        this->volume = (uint8_t*)realloc(this->volume, volume_lengthT * sizeof(uint8_t));
      offset += 3;
      volume_length = volume_lengthT;
      for( uint8_t i = 0; i < volume_length; i++){
      this->st_volume =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->st_volume);
        memcpy( &(this->volume[i]), &(this->st_volume), sizeof(uint8_t));
      }
     return offset;
    }

    const char * getType(){ return "supra_msgs/UInt8Image"; };
    const char * getMD5(){ return "ffca291c3c4d8f243ed044176fcd617b"; };

  };

}
#endif