#ifndef _ROS_smach_msgs_SmachContainerInitialStatusCmd_h
#define _ROS_smach_msgs_SmachContainerInitialStatusCmd_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace smach_msgs
{

  class SmachContainerInitialStatusCmd : public ros::Msg
  {
    public:
      const char* path;
      uint8_t initial_states_length;
      char* st_initial_states;
      char* * initial_states;
      const char* local_data;

    SmachContainerInitialStatusCmd():
      path(""),
      initial_states_length(0), initial_states(NULL),
      local_data("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
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
      uint32_t length_local_data = strlen(this->local_data);
      memcpy(outbuffer + offset, &length_local_data, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->local_data, length_local_data);
      offset += length_local_data;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
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
      uint32_t length_local_data;
      memcpy(&length_local_data, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_local_data; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_local_data-1]=0;
      this->local_data = (char *)(inbuffer + offset-1);
      offset += length_local_data;
     return offset;
    }

    const char * getType(){ return "smach_msgs/SmachContainerInitialStatusCmd"; };
    const char * getMD5(){ return "45f8cf31fc29b829db77f23001f788d6"; };

  };

}
#endif