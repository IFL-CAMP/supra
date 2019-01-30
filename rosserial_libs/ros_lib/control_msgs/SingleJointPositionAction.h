#ifndef _ROS_control_msgs_SingleJointPositionAction_h
#define _ROS_control_msgs_SingleJointPositionAction_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "control_msgs/SingleJointPositionActionGoal.h"
#include "control_msgs/SingleJointPositionActionResult.h"
#include "control_msgs/SingleJointPositionActionFeedback.h"

namespace control_msgs
{

  class SingleJointPositionAction : public ros::Msg
  {
    public:
      control_msgs::SingleJointPositionActionGoal action_goal;
      control_msgs::SingleJointPositionActionResult action_result;
      control_msgs::SingleJointPositionActionFeedback action_feedback;

    SingleJointPositionAction():
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

    const char * getType(){ return "control_msgs/SingleJointPositionAction"; };
    const char * getMD5(){ return "c4a786b7d53e5d0983decf967a5a779e"; };

  };

}
#endif