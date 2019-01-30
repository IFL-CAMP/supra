#ifndef _ROS_controller_manager_msgs_ControllerStatistics_h
#define _ROS_controller_manager_msgs_ControllerStatistics_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "ros/time.h"
#include "ros/duration.h"

namespace controller_manager_msgs
{

  class ControllerStatistics : public ros::Msg
  {
    public:
      const char* name;
      const char* type;
      ros::Time timestamp;
      bool running;
      ros::Duration max_time;
      ros::Duration mean_time;
      ros::Duration variance_time;
      int32_t num_control_loop_overruns;
      ros::Time time_last_control_loop_overrun;

    ControllerStatistics():
      name(""),
      type(""),
      timestamp(),
      running(0),
      max_time(),
      mean_time(),
      variance_time(),
      num_control_loop_overruns(0),
      time_last_control_loop_overrun()
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
      uint32_t length_type = strlen(this->type);
      memcpy(outbuffer + offset, &length_type, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->type, length_type);
      offset += length_type;
      *(outbuffer + offset + 0) = (this->timestamp.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->timestamp.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->timestamp.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->timestamp.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->timestamp.sec);
      *(outbuffer + offset + 0) = (this->timestamp.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->timestamp.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->timestamp.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->timestamp.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->timestamp.nsec);
      union {
        bool real;
        uint8_t base;
      } u_running;
      u_running.real = this->running;
      *(outbuffer + offset + 0) = (u_running.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->running);
      *(outbuffer + offset + 0) = (this->max_time.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->max_time.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->max_time.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->max_time.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->max_time.sec);
      *(outbuffer + offset + 0) = (this->max_time.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->max_time.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->max_time.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->max_time.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->max_time.nsec);
      *(outbuffer + offset + 0) = (this->mean_time.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->mean_time.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->mean_time.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->mean_time.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->mean_time.sec);
      *(outbuffer + offset + 0) = (this->mean_time.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->mean_time.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->mean_time.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->mean_time.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->mean_time.nsec);
      *(outbuffer + offset + 0) = (this->variance_time.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->variance_time.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->variance_time.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->variance_time.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->variance_time.sec);
      *(outbuffer + offset + 0) = (this->variance_time.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->variance_time.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->variance_time.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->variance_time.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->variance_time.nsec);
      union {
        int32_t real;
        uint32_t base;
      } u_num_control_loop_overruns;
      u_num_control_loop_overruns.real = this->num_control_loop_overruns;
      *(outbuffer + offset + 0) = (u_num_control_loop_overruns.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_num_control_loop_overruns.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_num_control_loop_overruns.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_num_control_loop_overruns.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->num_control_loop_overruns);
      *(outbuffer + offset + 0) = (this->time_last_control_loop_overrun.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->time_last_control_loop_overrun.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->time_last_control_loop_overrun.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->time_last_control_loop_overrun.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->time_last_control_loop_overrun.sec);
      *(outbuffer + offset + 0) = (this->time_last_control_loop_overrun.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->time_last_control_loop_overrun.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->time_last_control_loop_overrun.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->time_last_control_loop_overrun.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->time_last_control_loop_overrun.nsec);
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
      uint32_t length_type;
      memcpy(&length_type, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_type; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_type-1]=0;
      this->type = (char *)(inbuffer + offset-1);
      offset += length_type;
      this->timestamp.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->timestamp.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->timestamp.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->timestamp.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->timestamp.sec);
      this->timestamp.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->timestamp.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->timestamp.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->timestamp.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->timestamp.nsec);
      union {
        bool real;
        uint8_t base;
      } u_running;
      u_running.base = 0;
      u_running.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->running = u_running.real;
      offset += sizeof(this->running);
      this->max_time.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->max_time.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->max_time.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->max_time.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->max_time.sec);
      this->max_time.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->max_time.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->max_time.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->max_time.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->max_time.nsec);
      this->mean_time.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->mean_time.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->mean_time.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->mean_time.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->mean_time.sec);
      this->mean_time.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->mean_time.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->mean_time.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->mean_time.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->mean_time.nsec);
      this->variance_time.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->variance_time.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->variance_time.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->variance_time.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->variance_time.sec);
      this->variance_time.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->variance_time.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->variance_time.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->variance_time.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->variance_time.nsec);
      union {
        int32_t real;
        uint32_t base;
      } u_num_control_loop_overruns;
      u_num_control_loop_overruns.base = 0;
      u_num_control_loop_overruns.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_num_control_loop_overruns.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_num_control_loop_overruns.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_num_control_loop_overruns.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->num_control_loop_overruns = u_num_control_loop_overruns.real;
      offset += sizeof(this->num_control_loop_overruns);
      this->time_last_control_loop_overrun.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->time_last_control_loop_overrun.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->time_last_control_loop_overrun.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->time_last_control_loop_overrun.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->time_last_control_loop_overrun.sec);
      this->time_last_control_loop_overrun.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->time_last_control_loop_overrun.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->time_last_control_loop_overrun.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->time_last_control_loop_overrun.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->time_last_control_loop_overrun.nsec);
     return offset;
    }

    const char * getType(){ return "controller_manager_msgs/ControllerStatistics"; };
    const char * getMD5(){ return "697780c372c8d8597a1436d0e2ad3ba8"; };

  };

}
#endif