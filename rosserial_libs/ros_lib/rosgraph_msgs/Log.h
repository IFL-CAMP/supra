#ifndef _ROS_rosgraph_msgs_Log_h
#define _ROS_rosgraph_msgs_Log_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"

namespace rosgraph_msgs
{

  class Log : public ros::Msg
  {
    public:
      std_msgs::Header header;
      int8_t level;
      const char* name;
      const char* msg;
      const char* file;
      const char* function;
      uint32_t line;
      uint8_t topics_length;
      char* st_topics;
      char* * topics;
      enum { DEBUG = 1  };
      enum { INFO = 2   };
      enum { WARN = 4   };
      enum { ERROR = 8  };
      enum { FATAL = 16  };

    Log():
      header(),
      level(0),
      name(""),
      msg(""),
      file(""),
      function(""),
      line(0),
      topics_length(0), topics(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      union {
        int8_t real;
        uint8_t base;
      } u_level;
      u_level.real = this->level;
      *(outbuffer + offset + 0) = (u_level.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->level);
      uint32_t length_name = strlen(this->name);
      memcpy(outbuffer + offset, &length_name, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->name, length_name);
      offset += length_name;
      uint32_t length_msg = strlen(this->msg);
      memcpy(outbuffer + offset, &length_msg, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->msg, length_msg);
      offset += length_msg;
      uint32_t length_file = strlen(this->file);
      memcpy(outbuffer + offset, &length_file, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->file, length_file);
      offset += length_file;
      uint32_t length_function = strlen(this->function);
      memcpy(outbuffer + offset, &length_function, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->function, length_function);
      offset += length_function;
      *(outbuffer + offset + 0) = (this->line >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->line >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->line >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->line >> (8 * 3)) & 0xFF;
      offset += sizeof(this->line);
      *(outbuffer + offset++) = topics_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < topics_length; i++){
      uint32_t length_topicsi = strlen(this->topics[i]);
      memcpy(outbuffer + offset, &length_topicsi, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->topics[i], length_topicsi);
      offset += length_topicsi;
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      union {
        int8_t real;
        uint8_t base;
      } u_level;
      u_level.base = 0;
      u_level.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->level = u_level.real;
      offset += sizeof(this->level);
      uint32_t length_name;
      memcpy(&length_name, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_name; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_name-1]=0;
      this->name = (char *)(inbuffer + offset-1);
      offset += length_name;
      uint32_t length_msg;
      memcpy(&length_msg, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_msg; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_msg-1]=0;
      this->msg = (char *)(inbuffer + offset-1);
      offset += length_msg;
      uint32_t length_file;
      memcpy(&length_file, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_file; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_file-1]=0;
      this->file = (char *)(inbuffer + offset-1);
      offset += length_file;
      uint32_t length_function;
      memcpy(&length_function, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_function; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_function-1]=0;
      this->function = (char *)(inbuffer + offset-1);
      offset += length_function;
      this->line =  ((uint32_t) (*(inbuffer + offset)));
      this->line |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->line |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->line |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->line);
      uint8_t topics_lengthT = *(inbuffer + offset++);
      if(topics_lengthT > topics_length)
        this->topics = (char**)realloc(this->topics, topics_lengthT * sizeof(char*));
      offset += 3;
      topics_length = topics_lengthT;
      for( uint8_t i = 0; i < topics_length; i++){
      uint32_t length_st_topics;
      memcpy(&length_st_topics, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_topics; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_topics-1]=0;
      this->st_topics = (char *)(inbuffer + offset-1);
      offset += length_st_topics;
        memcpy( &(this->topics[i]), &(this->st_topics), sizeof(char*));
      }
     return offset;
    }

    const char * getType(){ return "rosgraph_msgs/Log"; };
    const char * getMD5(){ return "acffd30cd6b6de30f120938c17c593fb"; };

  };

}
#endif