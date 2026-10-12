#pragma once

#include "motion_controller.hpp"
#include "mpc_pipeline_client.hpp"
#include "basic_function.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "drdds/msg/imu_data.hpp"
#include "drdds/msg/joints_data.hpp"
#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <limits>

namespace q {
// Common acquisition and MPC path for normal and residual controllers.
class MPXController : public MotionController {
protected:
    static constexpr float kJointKp = 30.f;
    static constexpr float kJointKd = 1.f;
private:
    using Clock = std::chrono::steady_clock;
    using Array = std_msgs::msg::Float64MultiArray;
    std::unique_ptr<lite3_mpc::MPCPipelineClient> pipeline_;
    double epoch_ = 0.;
    rclcpp::Subscription<Array>::SharedPtr base_sub_;
    rclcpp::Subscription<drdds::msg::ImuData>::SharedPtr imu_sub_;
    rclcpp::Subscription<drdds::msg::JointsData>::SharedPtr joints_sub_;
    std::mutex mutex_;
    RobotBasicState observation_{};
    std::array<double, 7> base_{}; // timestamp, xyz, linear velocity
    double imu_stamp_ = 0, joints_stamp_ = 0;
    Clock::time_point base_received_, imu_received_, joints_received_;
    Clock::time_point entered_, requested_, completed_;
    bool have_base_ = false, have_imu_ = false, have_joints_ = false;
    bool active_ = false, failed_ = false;
    double requested_stamp_ = -std::numeric_limits<double>::infinity();
    MatXf output_ = MatXf::Zero(12, 5);

protected:
    // The FSM combines the latest asynchronous WBC output with any residual action.
    virtual std::chrono::milliseconds InferenceInterval() const { return std::chrono::milliseconds(5); }
    virtual void ResetPolicy() {}
    virtual MatXf ComputeCommand(const lite3_mpc::WBCOutput& mpc,
                                const RobotBasicState&, const UserCommand&, double) {
        RobotAction command;
        command.tau_ff = mpc.torque;
        command.goal_joint_pos = mpc.q_des;
        command.goal_joint_vel = mpc.dq_des;
        command.kp = VecXf::Constant(12,kJointKp);
        command.kd = VecXf::Constant(12,kJointKd);
        return command.ConvertToMat();
    }

public:
    MPXController(RobotName robot, std::shared_ptr<ControllerData> data)
        : MotionController(robot, data) {
        if (robot != RobotName::Lite3) throw std::invalid_argument("MPC supports Lite3 only");
        observation_.joint_pos = observation_.joint_vel = VecXf::Zero(12);
        auto node = ri_ptr_->get_node();
        pipeline_ = std::make_unique<lite3_mpc::MPCPipelineClient>(node);
        base_sub_ = node->create_subscription<Array>("/BASE_DATA", 1, [this](Array::SharedPtr msg) {
            if (msg->data.size() != 7) return;
            for (double x : msg->data) if (!std::isfinite(x)) return;
            std::lock_guard<std::mutex> lock(mutex_);
            std::copy(msg->data.begin(), msg->data.end(), base_.begin());
            base_received_ = Clock::now();
            have_base_ = true;
        });
        imu_sub_ = node->create_subscription<drdds::msg::ImuData>(
            "/IMU_DATA", rclcpp::SensorDataQoS().keep_last(1),
            [this](drdds::msg::ImuData::SharedPtr msg) {
                const auto& imu = msg->data;
                const Vec3f rpy(Deg2Rad(imu.roll), Deg2Rad(imu.pitch), Deg2Rad(imu.yaw));
                const Vec3f omega(imu.omega_x, imu.omega_y, imu.omega_z);
                if (!rpy.allFinite() || !omega.allFinite()) return;
                std::lock_guard<std::mutex> lock(mutex_);
                observation_.base_rpy = rpy;
                observation_.base_rot_mat = RpyToRm(rpy);
                observation_.base_omega = omega;
                imu_stamp_ = rclcpp::Time(msg->header.stamp).seconds();
                imu_received_ = Clock::now();
                have_imu_ = true;
            });
        joints_sub_ = node->create_subscription<drdds::msg::JointsData>(
            "/JOINTS_DATA", 1, [this](drdds::msg::JointsData::SharedPtr msg) {
                for (int i = 0; i < 12; ++i)
                    if (!std::isfinite(msg->data.joints_data[i].position) ||
                        !std::isfinite(msg->data.joints_data[i].velocity)) return;
                std::lock_guard<std::mutex> lock(mutex_);
                for (int i = 0; i < 12; ++i) {
                    observation_.joint_pos(i) = msg->data.joints_data[i].position;
                    observation_.joint_vel(i) = msg->data.joints_data[i].velocity;
                }
                joints_stamp_ = rclcpp::Time(msg->header.stamp).seconds();
                joints_received_ = Clock::now();
                have_joints_ = true;
            });
    }
    ~MPXController() override { Stop(); }
    void Compile() override {
        try { pipeline_->Start(cp_ptr_->torque_limit_); }
        catch(const std::exception& error){std::cerr << "[MPC] Initialization failed: " << error.what() << '\n';}
    }
    void Stop() override {
        {std::lock_guard<std::mutex> lock(mutex_);active_=false;}
        if(pipeline_) { pipeline_->Deactivate(epoch_); pipeline_.reset(); }
    }
    bool IsReady() override { return pipeline_ && pipeline_->IsReady(); }

    void OnEnter() override {
        output_.setZero();
        output_.col(1) = ri_ptr_->GetJointPosition();
        output_.col(0).setConstant(kJointKp);
        output_.col(2).setConstant(kJointKd);
        ri_ptr_->SetJointCommand(output_);
        ResetPolicy();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_ = true;
            failed_ = false;
            epoch_ = pipeline_->Activate(*uc_ptr_->GetUserCommand());
            entered_ = Clock::now();
            requested_ = Clock::time_point{};
            requested_stamp_ = -std::numeric_limits<double>::infinity();
            completed_ = Clock::time_point{};
        }
    }

    void OnExit() override {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
        if(pipeline_) pipeline_->Deactivate(epoch_);
    }

    void Run() override {
        const auto now = Clock::now();
        RobotBasicState snapshot;
        std::array<double, 7> base;
        const UserCommand command = *uc_ptr_->GetUserCommand();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!active_ || failed_ || !pipeline_) return;
            pipeline_->Update(epoch_,command);
            const auto timeout = std::chrono::milliseconds(150);
            const bool coherent = have_base_ && have_imu_ && have_joints_ &&
                now - base_received_ < timeout && now - imu_received_ < timeout &&
                now - joints_received_ < timeout && std::abs(base_[0] - imu_stamp_) <= .01 &&
                std::abs(base_[0] - joints_stamp_) <= .01;
            // Schedule on telemetry time: wall-clock callback jitter must not drop an update.
            // A repeated timestamp is held; a rewind starts a new backend epoch.
            const double interval = std::chrono::duration<double>(InferenceInterval()).count();
            if (!active_ || failed_ || !coherent ||
                (base_[0] >= requested_stamp_ && base_[0] - requested_stamp_ < interval - 1e-6)) return;
            if(base_[0] < requested_stamp_) {
                epoch_ = pipeline_->Activate(command);
                ResetPolicy();
            }
            snapshot = observation_;
            base = base_;
            requested_ = now;
            requested_stamp_ = base[0];
        }
        try {
            // No wait for either process. WBC uses its latest completed MPC plan;
            // residual ONNX inference overlaps with both independent processes.
            const auto latest = pipeline_->Latest(epoch_,base[0]);
            if (!latest) return;
            MatXf combined = ComputeCommand(*latest, snapshot, command, base[0]);
            std::lock_guard<std::mutex> lock(mutex_);
            if (!active_) return;
            if (Clock::now() - requested_ >= std::chrono::milliseconds(150))
                throw std::runtime_error("MPC control cycle timed out");
            output_ = std::move(combined);
            if (output_.rows() != 12 || output_.cols() != 5 || !output_.allFinite())
                throw std::runtime_error("Invalid combined joint command");
            completed_ = Clock::now();
            ri_ptr_->SetJointCommand(output_);
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
            std::cerr << "[MPC] " << error.what() << '\n';
        }
    }

    bool LoseControlJudge() override {
        if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = Clock::now();
        const bool stale = completed_ != Clock::time_point{} ? now - completed_ > std::chrono::milliseconds(150)
                                               : now - entered_ > std::chrono::seconds(2);
        if (failed_ || stale) std::cerr << "[MPC] Control cycle unavailable; joint damping.\n";
        return failed_ || stale;
    }
};
}
