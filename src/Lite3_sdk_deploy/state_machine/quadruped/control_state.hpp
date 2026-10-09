#pragma once

#include "state_base.h"
#include "rl_controller.hpp"
#include "mpx_controller.hpp"
#include <vector>
#include <stdexcept>

namespace q {
class ControlState : public StateBase {
public:
    struct ControllerEntry {
        const char* name;
        std::shared_ptr<MotionController> controller;
    };
private:
    // Add a controller here; the FSM and input interfaces do not need new states.
    std::vector<ControllerEntry> controllers_;
    size_t selected_ = 0;
    bool active_ = false;
public:
    ControlState(RobotName robot, std::shared_ptr<ControllerData> data)
        : StateBase(robot, "control", data), controllers_{
            {"rl_controller", std::make_shared<RLController>(robot, data)},
            {"mpx", std::make_shared<MPXController>(robot, data)},
        } {}

    const std::vector<ControllerEntry>& Controllers() const { return controllers_; }

    // The first selection from standing starts at the first registered controller.
    bool SelectNext() {
        const size_t next = active_ ? (selected_ + 1) % controllers_.size() : 0;
        auto& candidate = controllers_[next];
        if (!candidate.controller->IsReady()) {
            std::cout << "[CONTROLLER] " << candidate.name
                      << " is not ready: compilation is in progress or the worker is unavailable. "
                         "Request discarded; state unchanged. Press C / A after MPC READY."
                      << std::endl;
            return false;
        }
        if (active_) controllers_[selected_].controller->OnExit();
        selected_ = next;
        if (active_) candidate.controller->OnEnter();
        std::cout << "[CONTROLLER] Selected: " << candidate.name << std::endl;
        return true;
    }
    bool IsReady() { return controllers_[selected_].controller->IsReady(); }
    std::string ActiveControllerName() const {
        return active_ ? controllers_[selected_].name : "";
    }
    void OnEnter() override {
        if (!IsReady()) throw std::runtime_error("Selected controller is not ready");
        controllers_[selected_].controller->OnEnter();
        active_ = true;
        msfb_.UpdateCurrentState(RobotMotionState::ControlMode);
    }
    void OnExit() override {
        if (active_) controllers_[selected_].controller->OnExit();
        active_ = false;
        selected_ = 0;
    }
    void Run() override { controllers_[selected_].controller->Run(); }
    bool LoseControlJudge() override {
        const auto command = *uc_ptr_->GetUserCommand();
        return command.safe_control_mode != 0 ||
               command.target_mode == uint8_t(RobotMotionState::JointDamping) ||
               controllers_[selected_].controller->LoseControlJudge();
    }
    StateName GetNextStateName() override {
        const auto command = *uc_ptr_->GetUserCommand();
        if (command.safe_control_mode != 0 || command.target_mode == uint8_t(JointDamping))
            return kJointDamping;
        if (command.target_mode == uint8_t(LieDown)) return kLieDown;
        return kControl;
    }
};
}
