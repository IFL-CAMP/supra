#ifndef _ROS_SERVICE_GetModelProperties_h
#define _ROS_SERVICE_GetModelProperties_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace gazebo_msgs
{

static const char GETMODELPROPERTIES[] = "gazebo_msgs/GetModelProperties";

  class GetModelPropertiesRequest : public ros::Msg
  {
    public:
      const char* model_name;

    GetModelPropertiesRequest():
      model_name("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_model_name = strlen(this->model_name);
      memcpy(outbuffer + offset, &length_model_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->model_name, length_model_name);
      offset += length_model_name;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_model_name;
      memcpy(&length_model_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_model_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_model_name-1]=0;
      this->model_name = (char *)(inbuffer + offset-1);
      offset += length_model_name;
     return offset;
    }

    const char * getType(){ return GETMODELPROPERTIES; };
    const char * getMD5(){ return "ea31c8eab6fc401383cf528a7c0984ba"; };

  };

  class GetModelPropertiesResponse : public ros::Msg
  {
    public:
      const char* parent_model_name;
      const char* canonical_body_name;
      uint8_t body_names_length;
      char* st_body_names;
      char* * body_names;
      uint8_t geom_names_length;
      char* st_geom_names;
      char* * geom_names;
      uint8_t joint_names_length;
      char* st_joint_names;
      char* * joint_names;
      uint8_t child_model_names_length;
      char* st_child_model_names;
      char* * child_model_names;
      bool is_static;
      bool success;
      const char* status_message;

    GetModelPropertiesResponse():
      parent_model_name(""),
      canonical_body_name(""),
      body_names_length(0), body_names(NULL),
      geom_names_length(0), geom_names(NULL),
      joint_names_length(0), joint_names(NULL),
      child_model_names_length(0), child_model_names(NULL),
      is_static(0),
      success(0),
      status_message("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_parent_model_name = strlen(this->parent_model_name);
      memcpy(outbuffer + offset, &length_parent_model_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->parent_model_name, length_parent_model_name);
      offset += length_parent_model_name;
      uint32_t length_canonical_body_name = strlen(this->canonical_body_name);
      memcpy(outbuffer + offset, &length_canonical_body_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->canonical_body_name, length_canonical_body_name);
      offset += length_canonical_body_name;
      *(outbuffer + offset++) = body_names_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < body_names_length; i++){
      uint32_t length_body_namesi = strlen(this->body_names[i]);
      memcpy(outbuffer + offset, &length_body_namesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->body_names[i], length_body_namesi);
      offset += length_body_namesi;
      }
      *(outbuffer + offset++) = geom_names_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < geom_names_length; i++){
      uint32_t length_geom_namesi = strlen(this->geom_names[i]);
      memcpy(outbuffer + offset, &length_geom_namesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->geom_names[i], length_geom_namesi);
      offset += length_geom_namesi;
      }
      *(outbuffer + offset++) = joint_names_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < joint_names_length; i++){
      uint32_t length_joint_namesi = strlen(this->joint_names[i]);
      memcpy(outbuffer + offset, &length_joint_namesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->joint_names[i], length_joint_namesi);
      offset += length_joint_namesi;
      }
      *(outbuffer + offset++) = child_model_names_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < child_model_names_length; i++){
      uint32_t length_child_model_namesi = strlen(this->child_model_names[i]);
      memcpy(outbuffer + offset, &length_child_model_namesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->child_model_names[i], length_child_model_namesi);
      offset += length_child_model_namesi;
      }
      union {
        bool real;
        uint8_t base;
      } u_is_static;
      u_is_static.real = this->is_static;
      *(outbuffer + offset + 0) = (u_is_static.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->is_static);
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.real = this->success;
      *(outbuffer + offset + 0) = (u_success.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->success);
      uint32_t length_status_message = strlen(this->status_message);
      memcpy(outbuffer + offset, &length_status_message, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->status_message, length_status_message);
      offset += length_status_message;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_parent_model_name;
      memcpy(&length_parent_model_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_parent_model_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_parent_model_name-1]=0;
      this->parent_model_name = (char *)(inbuffer + offset-1);
      offset += length_parent_model_name;
      uint32_t length_canonical_body_name;
      memcpy(&length_canonical_body_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_canonical_body_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_canonical_body_name-1]=0;
      this->canonical_body_name = (char *)(inbuffer + offset-1);
      offset += length_canonical_body_name;
      uint8_t body_names_lengthT = *(inbuffer + offset++);
      if(body_names_lengthT > body_names_length)
        this->body_names = (char**)realloc(this->body_names, body_names_lengthT * sizeof(char*));
      offset += 3;
      body_names_length = body_names_lengthT;
      for( uint8_t i = 0; i < body_names_length; i++){
      uint32_t length_st_body_names;
      memcpy(&length_st_body_names, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_body_names; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_body_names-1]=0;
      this->st_body_names = (char *)(inbuffer + offset-1);
      offset += length_st_body_names;
        memcpy( &(this->body_names[i]), &(this->st_body_names), sizeof(char*));
      }
      uint8_t geom_names_lengthT = *(inbuffer + offset++);
      if(geom_names_lengthT > geom_names_length)
        this->geom_names = (char**)realloc(this->geom_names, geom_names_lengthT * sizeof(char*));
      offset += 3;
      geom_names_length = geom_names_lengthT;
      for( uint8_t i = 0; i < geom_names_length; i++){
      uint32_t length_st_geom_names;
      memcpy(&length_st_geom_names, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_geom_names; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_geom_names-1]=0;
      this->st_geom_names = (char *)(inbuffer + offset-1);
      offset += length_st_geom_names;
        memcpy( &(this->geom_names[i]), &(this->st_geom_names), sizeof(char*));
      }
      uint8_t joint_names_lengthT = *(inbuffer + offset++);
      if(joint_names_lengthT > joint_names_length)
        this->joint_names = (char**)realloc(this->joint_names, joint_names_lengthT * sizeof(char*));
      offset += 3;
      joint_names_length = joint_names_lengthT;
      for( uint8_t i = 0; i < joint_names_length; i++){
      uint32_t length_st_joint_names;
      memcpy(&length_st_joint_names, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_joint_names; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_joint_names-1]=0;
      this->st_joint_names = (char *)(inbuffer + offset-1);
      offset += length_st_joint_names;
        memcpy( &(this->joint_names[i]), &(this->st_joint_names), sizeof(char*));
      }
      uint8_t child_model_names_lengthT = *(inbuffer + offset++);
      if(child_model_names_lengthT > child_model_names_length)
        this->child_model_names = (char**)realloc(this->child_model_names, child_model_names_lengthT * sizeof(char*));
      offset += 3;
      child_model_names_length = child_model_names_lengthT;
      for( uint8_t i = 0; i < child_model_names_length; i++){
      uint32_t length_st_child_model_names;
      memcpy(&length_st_child_model_names, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_child_model_names; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_child_model_names-1]=0;
      this->st_child_model_names = (char *)(inbuffer + offset-1);
      offset += length_st_child_model_names;
        memcpy( &(this->child_model_names[i]), &(this->st_child_model_names), sizeof(char*));
      }
      union {
        bool real;
        uint8_t base;
      } u_is_static;
      u_is_static.base = 0;
      u_is_static.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->is_static = u_is_static.real;
      offset += sizeof(this->is_static);
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.base = 0;
      u_success.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->success = u_success.real;
      offset += sizeof(this->success);
      uint32_t length_status_message;
      memcpy(&length_status_message, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_status_message; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_status_message-1]=0;
      this->status_message = (char *)(inbuffer + offset-1);
      offset += length_status_message;
     return offset;
    }

    const char * getType(){ return GETMODELPROPERTIES; };
    const char * getMD5(){ return "b7f370938ef77b464b95f1bab3ec5028"; };

  };

  class GetModelProperties {
    public:
    typedef GetModelPropertiesRequest Request;
    typedef GetModelPropertiesResponse Response;
  };

}
#endif
