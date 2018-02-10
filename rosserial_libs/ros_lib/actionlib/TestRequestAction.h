#ifndef _ROS_actionlib_TestRequestAction_h
#define _ROS_actionlib_TestRequestAction_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "actionlib/TestRequestActionGoal.h"
#include "actionlib/TestRequestActionResult.h"
#include "actionlib/TestRequestActionFeedback.h"

namespace actionlib
{

  class TestRequestAction : public ros::Msg
  {
    public:
      actionlib::TestRequestActionGoal action_goal;
      actionlib::TestRequestActionResult action_result;
      actionlib::TestRequestActionFeedback action_feedback;

    TestRequestAction():
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

    const char * getType(){ return "actionlib/TestRequestAction"; };
    const char * getMD5(){ return "dc44b1f4045dbf0d1db54423b3b86b30"; };

  };

}
#endif