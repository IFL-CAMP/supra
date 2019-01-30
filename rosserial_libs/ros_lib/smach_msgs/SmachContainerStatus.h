#ifndef _ROS_smach_msgs_SmachContainerStatus_h
#define _ROS_smach_msgs_SmachContainerStatus_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"

namespace smach_msgs
{

  class SmachContainerStatus : public ros::Msg
  {
    public:
      std_msgs::Header header;
      const char* path;
      uint8_t initial_states_length;
      char* st_initial_states;
      char* * initial_states;
      uint8_t active_states_length;
      char* st_active_states;
      char* * active_states;
      const char* local_data;
      const char* info;

    SmachContainerStatus():
      header(),
      path(""),
      initial_states_length(0), initial_states(NULL),
      active_states_length(0), active_states(NULL),
      local_data(""),
      info("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      uint32_t length_path = strlen(this->path);
      memcpy(outbuffer + offset, &length_path, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->path, length_path);
      offset += length_path;
      *(outbuffer + offset++) = initial_states_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < initial_states_length; i++){
      uint32_t length_initial_statesi = strlen(this->initial_states[i]);
      memcpy(outbuffer + offset, &length_initial_statesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->initial_states[i], length_initial_statesi);
      offset += length_initial_statesi;
      }
      *(outbuffer + offset++) = active_states_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < active_states_length; i++){
      uint32_t length_active_statesi = strlen(this->active_states[i]);
      memcpy(outbuffer + offset, &length_active_statesi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->active_states[i], length_active_statesi);
      offset += length_active_statesi;
      }
      uint32_t length_local_data = strlen(this->local_data);
      memcpy(outbuffer + offset, &length_local_data, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->local_data, length_local_data);
      offset += length_local_data;
      uint32_t length_info = strlen(this->info);
      memcpy(outbuffer + offset, &length_info, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->info, length_info);
      offset += length_info;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      uint32_t length_path;
      memcpy(&length_path, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_path; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_path-1]=0;
      this->path = (char *)(inbuffer + offset-1);
      offset += length_path;
      uint8_t initial_states_lengthT = *(inbuffer + offset++);
      if(initial_states_lengthT > initial_states_length)
        this->initial_states = (char**)realloc(this->initial_states, initial_states_lengthT * sizeof(char*));
      offset += 3;
      initial_states_length = initial_states_lengthT;
      for( uint8_t i = 0; i < initial_states_length; i++){
      uint32_t length_st_initial_states;
      memcpy(&length_st_initial_states, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_initial_states; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_initial_states-1]=0;
      this->st_initial_states = (char *)(inbuffer + offset-1);
      offset += length_st_initial_states;
        memcpy( &(this->initial_states[i]), &(this->st_initial_states), sizeof(char*));
      }
      uint8_t active_states_lengthT = *(inbuffer + offset++);
      if(active_states_lengthT > active_states_length)
        this->active_states = (char**)realloc(this->active_states, active_states_lengthT * sizeof(char*));
      offset += 3;
      active_states_length = active_states_lengthT;
      for( uint8_t i = 0; i < active_states_length; i++){
      uint32_t length_st_active_states;
      memcpy(&length_st_active_states, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_active_states; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_active_states-1]=0;
      this->st_active_states = (char *)(inbuffer + offset-1);
      offset += length_st_active_states;
        memcpy( &(this->active_states[i]), &(this->st_active_states), sizeof(char*));
      }
      uint32_t length_local_data;
      memcpy(&length_local_data, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_local_data; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_local_data-1]=0;
      this->local_data = (char *)(inbuffer + offset-1);
      offset += length_local_data;
      uint32_t length_info;
      memcpy(&length_info, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_info; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_info-1]=0;
      this->info = (char *)(inbuffer + offset-1);
      offset += length_info;
     return offset;
    }

    const char * getType(){ return "smach_msgs/SmachContainerStatus"; };
    const char * getMD5(){ return "5ba2bb79ac19e3842d562a191f2a675b"; };

  };

}
#endif