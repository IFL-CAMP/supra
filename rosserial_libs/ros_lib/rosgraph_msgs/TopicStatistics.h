#ifndef _ROS_rosgraph_msgs_TopicStatistics_h
#define _ROS_rosgraph_msgs_TopicStatistics_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "ros/time.h"
#include "ros/duration.h"

namespace rosgraph_msgs
{

  class TopicStatistics : public ros::Msg
  {
    public:
      const char* topic;
      const char* node_pub;
      const char* node_sub;
      ros::Time window_start;
      ros::Time window_stop;
      int32_t delivered_msgs;
      int32_t dropped_msgs;
      int32_t traffic;
      ros::Duration period_mean;
      ros::Duration period_stddev;
      ros::Duration period_max;
      ros::Duration stamp_age_mean;
      ros::Duration stamp_age_stddev;
      ros::Duration stamp_age_max;

    TopicStatistics():
      topic(""),
      node_pub(""),
      node_sub(""),
      window_start(),
      window_stop(),
      delivered_msgs(0),
      dropped_msgs(0),
      traffic(0),
      period_mean(),
      period_stddev(),
      period_max(),
      stamp_age_mean(),
      stamp_age_stddev(),
      stamp_age_max()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_topic = strlen(this->topic);
      memcpy(outbuffer + offset, &length_topic, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->topic, length_topic);
      offset += length_topic;
      uint32_t length_node_pub = strlen(this->node_pub);
      memcpy(outbuffer + offset, &length_node_pub, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->node_pub, length_node_pub);
      offset += length_node_pub;
      uint32_t length_node_sub = strlen(this->node_sub);
      memcpy(outbuffer + offset, &length_node_sub, sizeof(uint32_t));
      offset += 4;
      memcpy(outbuffer + offset, this->node_sub, length_node_sub);
      offset += length_node_sub;
      *(outbuffer + offset + 0) = (this->window_start.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->window_start.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->window_start.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->window_start.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->window_start.sec);
      *(outbuffer + offset + 0) = (this->window_start.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->window_start.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->window_start.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->window_start.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->window_start.nsec);
      *(outbuffer + offset + 0) = (this->window_stop.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->window_stop.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->window_stop.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->window_stop.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->window_stop.sec);
      *(outbuffer + offset + 0) = (this->window_stop.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->window_stop.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->window_stop.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->window_stop.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->window_stop.nsec);
      union {
        int32_t real;
        uint32_t base;
      } u_delivered_msgs;
      u_delivered_msgs.real = this->delivered_msgs;
      *(outbuffer + offset + 0) = (u_delivered_msgs.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_delivered_msgs.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_delivered_msgs.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_delivered_msgs.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->delivered_msgs);
      union {
        int32_t real;
        uint32_t base;
      } u_dropped_msgs;
      u_dropped_msgs.real = this->dropped_msgs;
      *(outbuffer + offset + 0) = (u_dropped_msgs.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_dropped_msgs.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_dropped_msgs.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_dropped_msgs.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->dropped_msgs);
      union {
        int32_t real;
        uint32_t base;
      } u_traffic;
      u_traffic.real = this->traffic;
      *(outbuffer + offset + 0) = (u_traffic.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_traffic.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_traffic.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_traffic.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->traffic);
      *(outbuffer + offset + 0) = (this->period_mean.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->period_mean.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->period_mean.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->period_mean.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->period_mean.sec);
      *(outbuffer + offset + 0) = (this->period_mean.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->period_mean.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->period_mean.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->period_mean.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->period_mean.nsec);
      *(outbuffer + offset + 0) = (this->period_stddev.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->period_stddev.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->period_stddev.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->period_stddev.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->period_stddev.sec);
      *(outbuffer + offset + 0) = (this->period_stddev.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->period_stddev.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->period_stddev.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->period_stddev.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->period_stddev.nsec);
      *(outbuffer + offset + 0) = (this->period_max.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->period_max.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->period_max.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->period_max.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->period_max.sec);
      *(outbuffer + offset + 0) = (this->period_max.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->period_max.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->period_max.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->period_max.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->period_max.nsec);
      *(outbuffer + offset + 0) = (this->stamp_age_mean.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->stamp_age_mean.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->stamp_age_mean.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->stamp_age_mean.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->stamp_age_mean.sec);
      *(outbuffer + offset + 0) = (this->stamp_age_mean.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->stamp_age_mean.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->stamp_age_mean.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->stamp_age_mean.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->stamp_age_mean.nsec);
      *(outbuffer + offset + 0) = (this->stamp_age_stddev.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->stamp_age_stddev.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->stamp_age_stddev.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->stamp_age_stddev.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->stamp_age_stddev.sec);
      *(outbuffer + offset + 0) = (this->stamp_age_stddev.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->stamp_age_stddev.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->stamp_age_stddev.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->stamp_age_stddev.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->stamp_age_stddev.nsec);
      *(outbuffer + offset + 0) = (this->stamp_age_max.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->stamp_age_max.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->stamp_age_max.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->stamp_age_max.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->stamp_age_max.sec);
      *(outbuffer + offset + 0) = (this->stamp_age_max.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->stamp_age_max.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->stamp_age_max.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->stamp_age_max.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->stamp_age_max.nsec);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_topic;
      memcpy(&length_topic, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_topic; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_topic-1]=0;
      this->topic = (char *)(inbuffer + offset-1);
      offset += length_topic;
      uint32_t length_node_pub;
      memcpy(&length_node_pub, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_node_pub; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_node_pub-1]=0;
      this->node_pub = (char *)(inbuffer + offset-1);
      offset += length_node_pub;
      uint32_t length_node_sub;
      memcpy(&length_node_sub, (inbuffer + offset), sizeof(uint32_t));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_node_sub; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_node_sub-1]=0;
      this->node_sub = (char *)(inbuffer + offset-1);
      offset += length_node_sub;
      this->window_start.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->window_start.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->window_start.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->window_start.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->window_start.sec);
      this->window_start.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->window_start.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->window_start.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->window_start.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->window_start.nsec);
      this->window_stop.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->window_stop.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->window_stop.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->window_stop.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->window_stop.sec);
      this->window_stop.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->window_stop.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->window_stop.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->window_stop.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->window_stop.nsec);
      union {
        int32_t real;
        uint32_t base;
      } u_delivered_msgs;
      u_delivered_msgs.base = 0;
      u_delivered_msgs.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_delivered_msgs.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_delivered_msgs.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_delivered_msgs.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->delivered_msgs = u_delivered_msgs.real;
      offset += sizeof(this->delivered_msgs);
      union {
        int32_t real;
        uint32_t base;
      } u_dropped_msgs;
      u_dropped_msgs.base = 0;
      u_dropped_msgs.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_dropped_msgs.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_dropped_msgs.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_dropped_msgs.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->dropped_msgs = u_dropped_msgs.real;
      offset += sizeof(this->dropped_msgs);
      union {
        int32_t real;
        uint32_t base;
      } u_traffic;
      u_traffic.base = 0;
      u_traffic.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_traffic.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_traffic.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_traffic.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->traffic = u_traffic.real;
      offset += sizeof(this->traffic);
      this->period_mean.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->period_mean.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->period_mean.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->period_mean.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->period_mean.sec);
      this->period_mean.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->period_mean.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->period_mean.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->period_mean.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->period_mean.nsec);
      this->period_stddev.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->period_stddev.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->period_stddev.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->period_stddev.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->period_stddev.sec);
      this->period_stddev.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->period_stddev.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->period_stddev.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->period_stddev.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->period_stddev.nsec);
      this->period_max.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->period_max.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->period_max.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->period_max.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->period_max.sec);
      this->period_max.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->period_max.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->period_max.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->period_max.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->period_max.nsec);
      this->stamp_age_mean.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->stamp_age_mean.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->stamp_age_mean.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->stamp_age_mean.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->stamp_age_mean.sec);
      this->stamp_age_mean.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->stamp_age_mean.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->stamp_age_mean.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->stamp_age_mean.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->stamp_age_mean.nsec);
      this->stamp_age_stddev.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->stamp_age_stddev.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->stamp_age_stddev.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->stamp_age_stddev.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->stamp_age_stddev.sec);
      this->stamp_age_stddev.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->stamp_age_stddev.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->stamp_age_stddev.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->stamp_age_stddev.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->stamp_age_stddev.nsec);
      this->stamp_age_max.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->stamp_age_max.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->stamp_age_max.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->stamp_age_max.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->stamp_age_max.sec);
      this->stamp_age_max.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->stamp_age_max.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->stamp_age_max.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->stamp_age_max.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->stamp_age_max.nsec);
     return offset;
    }

    const char * getType(){ return "rosgraph_msgs/TopicStatistics"; };
    const char * getMD5(){ return "10152ed868c5097a5e2e4a89d7daa710"; };

  };

}
#endif