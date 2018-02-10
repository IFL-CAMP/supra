#ifndef _ROS_actionlib_tutorials_FibonacciActionFeedback_h
#define _ROS_actionlib_tutorials_FibonacciActionFeedback_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"
#include "actionlib_msgs/GoalStatus.h"
#include "actionlib_tutorials/FibonacciFeedback.h"

namespace actionlib_tutorials
{

  class FibonacciActionFeedback : public ros::Msg
  {
    public:
      std_msgs::Header header;
      actionlib_msgs::GoalStatus status;
      actionlib_tutorials::FibonacciFeedback feedback;

    FibonacciActionFeedback():
      header(),
      status(),
      feedback()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      offset += this->status.serialize(outbuffer + offset);
      offset += this->feedback.serialize(outbuffer + offset);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      offset += this->status.deserialize(inbuffer + offset);
      offset += this->feedback.deserialize(inbuffer + offset);
     return offset;
    }

    const char * getType(){ return "actionlib_tutorials/FibonacciActionFeedback"; };
    const char * getMD5(){ return "73b8497a9f629a31c0020900e4148f07"; };

  };

}
#endif