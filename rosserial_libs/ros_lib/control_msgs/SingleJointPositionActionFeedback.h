#ifndef _ROS_control_msgs_SingleJointPositionActionFeedback_h
#define _ROS_control_msgs_SingleJointPositionActionFeedback_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"
#include "actionlib_msgs/GoalStatus.h"
#include "control_msgs/SingleJointPositionFeedback.h"

namespace control_msgs
{

  class SingleJointPositionActionFeedback : public ros::Msg
  {
    public:
      std_msgs::Header header;
      actionlib_msgs::GoalStatus status;
      control_msgs::SingleJointPositionFeedback feedback;

    SingleJointPositionActionFeedback():
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

    const char * getType(){ return "control_msgs/SingleJointPositionActionFeedback"; };
    const char * getMD5(){ return "3503b7cf8972f90d245850a5d8796cfa"; };

  };

}
#endif