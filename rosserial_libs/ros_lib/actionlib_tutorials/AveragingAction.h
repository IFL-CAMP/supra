#ifndef _ROS_actionlib_tutorials_AveragingAction_h
#define _ROS_actionlib_tutorials_AveragingAction_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "actionlib_tutorials/AveragingActionGoal.h"
#include "actionlib_tutorials/AveragingActionResult.h"
#include "actionlib_tutorials/AveragingActionFeedback.h"

namespace actionlib_tutorials
{

  class AveragingAction : public ros::Msg
  {
    public:
      actionlib_tutorials::AveragingActionGoal action_goal;
      actionlib_tutorials::AveragingActionResult action_result;
      actionlib_tutorials::AveragingActionFeedback action_feedback;

    AveragingAction():
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

    const char * getType(){ return "actionlib_tutorials/AveragingAction"; };
    const char * getMD5(){ return "628678f2b4fa6a5951746a4a2d39e716"; };

  };

}
#endif