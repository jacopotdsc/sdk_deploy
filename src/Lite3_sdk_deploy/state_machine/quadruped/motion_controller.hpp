#pragma once

#include "state_base.h"

namespace q {
// Controllers own their computation; ControlState owns FSM transitions.
class MotionController {
protected:
    const RobotName robot_name_;
    std::shared_ptr<RobotInterface> ri_ptr_;
    std::shared_ptr<UserCommandInterface> uc_ptr_;
    std::shared_ptr<ControlParameters> cp_ptr_;
public:
    MotionController(RobotName robot, std::shared_ptr<ControllerData> data)
        : robot_name_(robot), ri_ptr_(data->ri_ptr), uc_ptr_(data->uc_ptr), cp_ptr_(data->cp_ptr) {}
    virtual ~MotionController() = default;
    // One-time resource preparation (e.g. background solver compilation).
    // Controllers without preparation work, such as RL, inherit these no-ops.
    virtual void Compile() {}
    virtual void Stop() {}
    virtual bool IsReady() = 0;
    virtual void OnEnter() = 0;
    virtual void Run() = 0;
    virtual void OnExit() = 0;
    virtual bool LoseControlJudge() = 0;
};
}
