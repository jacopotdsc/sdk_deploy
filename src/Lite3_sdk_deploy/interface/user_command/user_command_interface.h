/**
 * @file user_command_interface.h
 * @brief this file is used for robot's user command input
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once

#include "common_types.h"
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include "custom_types.h"
#include "motion_state_feedback.hpp"

using namespace types;

namespace interface{

class UserCommandInterface{
private:
    std::string active_controller_name_ = "";
    std::vector<std::string> available_controllers_;
protected:
    std::atomic<bool> controller_selection_requested_{false};
    std::mutex command_mutex_;
    UserCommand* CommandSnapshot() {
        thread_local UserCommand snapshot;
        std::lock_guard<std::mutex> lock(command_mutex_);
        snapshot = *usr_cmd_;
        return &snapshot;
    }


public:
    UserCommandInterface(RobotName robot_name){
        robot_name_ = robot_name;
        usr_cmd_ = new UserCommand();
        std::memset(usr_cmd_, 0, sizeof(UserCommand));
    }
    virtual ~UserCommandInterface(){
        delete usr_cmd_;
    }

    /**
     * @brief start the thread to process user command
     */
    virtual void Start() = 0;

    /**
     * @brief stop the thread 
     */
    virtual void Stop() = 0;

    /**
     * @brief return your user command
     * @return UserCommand ptr
     */
    virtual UserCommand* GetUserCommand() = 0; 

    /**
     * @brief set the motion state feedback 
     * @param  msfb         motion state feedback
     */
    virtual void SetMotionStateFeedback(MotionStateFeedback* msfb){
        msfb_ = msfb;
    }

    virtual void SetSafetyMode(uint8_t mode) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        usr_cmd_->safe_control_mode = mode;
    }
    virtual void SetTargetMode(uint8_t mode) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        usr_cmd_->target_mode = mode;
    }
    // Configure once before Start(), from the FSM's registered controller list.
    void SetAvailableControllers(const std::vector<std::string>& names) {
        available_controllers_ = names;
    }
    const std::vector<std::string>& GetAvailableControllers() const {
        return available_controllers_;
    }

    // Return the current controller, not an unvalidated future selection.
    std::string RequestNextController() {
        std::lock_guard<std::mutex> lock(command_mutex_);
        const std::string current = active_controller_name_;
        usr_cmd_->target_mode = uint8_t(RobotMotionState::ControlMode);
        controller_selection_requested_.store(true);
        return current;
    }
    void SetActiveControllerName(const std::string& name) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        active_controller_name_ = name.empty() ? "none" : name;
    }
    bool ConsumeControllerSelection() { return controller_selection_requested_.exchange(false); }

    MotionStateFeedback *msfb_ = nullptr;
    RobotName robot_name_;
    UserCommand *usr_cmd_;
};
};
