#ifndef _ROS_control_msgs_GripperCommandAction_h
#define _ROS_control_msgs_GripperCommandAction_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "control_msgs/GripperCommandActionGoal.h"
#include "control_msgs/GripperCommandActionResult.h"
#include "control_msgs/GripperCommandActionFeedback.h"

namespace control_msgs
{

  class GripperCommandAction : public ros::Msg
  {
    public:
      control_msgs::GripperCommandActionGoal action_goal;
      control_msgs::GripperCommandActionResult action_result;
      control_msgs::GripperCommandActionFeedback action_feedback;

    GripperCommandAction():
      action_goal(),
      action_result(),
      action_feedback()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->action_goal.serialize(outbuffer + offset);
      offset += this->action_result.serialize(outbuffer + offset);
      offset += this->action_feedback.serialize(outbuffer + offset);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->action_goal.deserialize(inbuffer + offset);
      offset += this->action_result.deserialize(inbuffer + offset);
      offset += this->action_feedback.deserialize(inbuffer + offset);
     return offset;
    }

    const char * getType(){ return "control_msgs/GripperCommandAction"; };
    const char * getMD5(){ return "950b2a6ebe831f5d4f4ceaba3d8be01e"; };

  };

}
#endif