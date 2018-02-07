#ifndef _ROS_turtle_actionlib_ShapeAction_h
#define _ROS_turtle_actionlib_ShapeAction_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "turtle_actionlib/ShapeActionGoal.h"
#include "turtle_actionlib/ShapeActionResult.h"
#include "turtle_actionlib/ShapeActionFeedback.h"

namespace turtle_actionlib
{

  class ShapeAction : public ros::Msg
  {
    public:
      turtle_actionlib::ShapeActionGoal action_goal;
      turtle_actionlib::ShapeActionResult action_result;
      turtle_actionlib::ShapeActionFeedback action_feedback;

    ShapeAction():
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

    const char * getType(){ return "turtle_actionlib/ShapeAction"; };
    const char * getMD5(){ return "d73b17d6237a925511f5d7727a1dc903"; };

  };

}
#endif