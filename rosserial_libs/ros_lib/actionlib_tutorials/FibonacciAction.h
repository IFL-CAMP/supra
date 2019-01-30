#ifndef _ROS_actionlib_tutorials_FibonacciAction_h
#define _ROS_actionlib_tutorials_FibonacciAction_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "actionlib_tutorials/FibonacciActionGoal.h"
#include "actionlib_tutorials/FibonacciActionResult.h"
#include "actionlib_tutorials/FibonacciActionFeedback.h"

namespace actionlib_tutorials
{

  class FibonacciAction : public ros::Msg
  {
    public:
      actionlib_tutorials::FibonacciActionGoal action_goal;
      actionlib_tutorials::FibonacciActionResult action_result;
      actionlib_tutorials::FibonacciActionFeedback action_feedback;

    FibonacciAction():
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

    const char * getType(){ return "actionlib_tutorials/FibonacciAction"; };
    const char * getMD5(){ return "f59df5767bf7634684781c92598b2406"; };

  };

}
#endif