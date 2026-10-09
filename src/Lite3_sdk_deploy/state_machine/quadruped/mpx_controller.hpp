#pragma once

#include "motion_controller.hpp"
#include "mpc_process.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <chrono>
#include <mutex>

namespace q {
// rl_deploy remains the only publisher of motor commands, including in MPC mode.
class MPXController : public MotionController {
    using Clock = std::chrono::steady_clock;
    MPCProcess worker_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr active_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr torque_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ready_sub_;
    bool ready_ = false;
    Clock::time_point ready_received_;
    std::mutex mutex_;
    VecXf torque_ = VecXf::Zero(12), position_ = VecXf::Zero(12), velocity_ = VecXf::Zero(12), hold_position_;
    Clock::time_point entered_, received_, published_;
    bool active_ = false, have_torque_ = false;
    void PublishActive(bool active) {
        std_msgs::msg::Bool msg;
        msg.data = active;
        active_pub_->publish(msg);
    }
public:
    MPXController(const RobotName& robot, std::shared_ptr<ControllerData> data) : MotionController(robot, data) {
        auto node = ri_ptr_->get_node();
        ready_sub_ = node->create_subscription<std_msgs::msg::Bool>(
            "lite3/mpc/ready", 1, [this](std_msgs::msg::Bool::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                ready_ = msg->data;
                ready_received_ = Clock::now();
            });
        active_pub_ = node->create_publisher<std_msgs::msg::Bool>("lite3/mpc/active", 1);
        velocity_pub_ = node->create_publisher<geometry_msgs::msg::Twist>("lite3/mpc/cmd_vel", 1);
        torque_sub_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
            "lite3/mpc/torque", 1, [this](std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!active_ || msg->data.size() != 36) return;
                for (double value : msg->data) if (!std::isfinite(value)) return;
                for (int i = 0; i < 12; ++i)
                    torque_(i) = LimitNumber(float(msg->data[i]), -cp_ptr_->torque_limit_(i%3), cp_ptr_->torque_limit_(i%3));
                for (int i = 0; i < 12; ++i) {
                    position_(i) = float(msg->data[12+i]);
                    velocity_(i) = float(msg->data[24+i]);
                }
                received_ = Clock::now();
                have_torque_ = true;
            });
    }
    ~MPXController() override { Stop(); }
    void Compile() override { worker_.Start(); }
    void Stop() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_ = false;
            active_ = false;
            have_torque_ = false;
        }
        worker_.Stop();
    }
    bool IsReady() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return ready_ && Clock::now()-ready_received_ < std::chrono::seconds(1);
    }
    void OnEnter() override {
        hold_position_ = ri_ptr_->GetJointPosition();
        { std::lock_guard<std::mutex> lock(mutex_);
          active_ = true; have_torque_ = false; entered_ = Clock::now(); }
        published_ = Clock::time_point{};
        PublishActive(true);
    }
    void OnExit() override {
        { std::lock_guard<std::mutex> lock(mutex_); active_ = false; have_torque_ = false; }
        PublishActive(false);
    }
    void Run() override {
        const auto now = Clock::now();
        if (now - published_ >= std::chrono::milliseconds(20)) {
            geometry_msgs::msg::Twist velocity;
            auto command = uc_ptr_->GetUserCommand();
            velocity.linear.x = command->forward_vel_scale;
            velocity.linear.y = command->side_vel_scale;
            velocity.angular.z = command->turnning_vel_scale;
            velocity_pub_->publish(velocity);
            PublishActive(true);
            published_ = now;
        }
        MatXf output = MatXf::Zero(12, 5);
        { std::lock_guard<std::mutex> lock(mutex_);
          if (have_torque_) {
              output.col(4) = torque_;
              // Use the same local joint PD gains as the Lite3 policy.
              output.col(0).setConstant(30.0f);
              output.col(2).setConstant(1.0f);
              output.col(1) = position_;
              output.col(3) = velocity_;
          }
          else {
              output.col(1) = hold_position_;
              for (int i=0; i<12; ++i) {
                  output(i,0) = cp_ptr_->swing_leg_kp_(i%3);
                  output(i,2) = cp_ptr_->swing_leg_kd_(i%3);
              }
          }
        }
        ri_ptr_->SetJointCommand(output);
    }
    bool LoseControlJudge() override {
        if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
        std::lock_guard<std::mutex> lock(mutex_);
        const bool stale = have_torque_ ? Clock::now()-received_ > std::chrono::milliseconds(150)
                                       : Clock::now()-entered_ > std::chrono::seconds(2);
        if (stale) std::cerr << "[MPC] MPC is not responding; switching to joint damping.\n";
        return stale;
    }
};
}
