/**
 * @file rl_controller.hpp
 * @brief rl policy runnning state for quadruped robot
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once
#include "motion_controller.hpp"
#include "policy_runner_base.hpp"
#include "lite3_policy_runner.hpp"
#include "robot_interface.h"
#include "user_command_interface.h"
#include "json.hpp"
#include "basic_function.hpp"
#include <atomic>
#include <mutex>

namespace q {
    class RLController : public MotionController {
    private:
        RobotBasicState rbs_;
        std::mutex observation_mutex_;
        int state_run_cnt_ = -1; // guarded by observation_mutex_ while the worker runs

        std::shared_ptr<PolicyRunnerBase> policy_ptr_;
        std::shared_ptr<Lite3PolicyRunner> lite3_policy_;

        std::thread run_policy_thread_;
        std::atomic<bool> start_flag_{false};

        double policy_cost_time_ = 1;

        Eigen::MatrixXf acc_rot = Eigen::MatrixXf::Zero(20, 3);
        int acc_rot_count = 0;

        void init_rbs_() {
            rbs_.flt_base_acc_mat = Eigen::MatrixXf::Zero(20, 3);
        }

        void UpdateRobotObservation() {
            rbs_.base_rpy = ri_ptr_->GetImuRpy();
            rbs_.base_rot_mat = RpyToRm(rbs_.base_rpy);
            rbs_.base_omega = ri_ptr_->GetImuOmega();
            rbs_.base_acc = ri_ptr_->GetImuAcc();
            rbs_.joint_pos = ri_ptr_->GetJointPosition();
            rbs_.joint_vel = ri_ptr_->GetJointVelocity();
            rbs_.joint_tau = ri_ptr_->GetJointTorque();

            // 储存
            rbs_.flt_base_acc_mat.row(acc_rot_count) = rbs_.base_acc.transpose();
            acc_rot_count += 1;
            acc_rot_count = acc_rot_count % 20;
        }

        void PolicyRunner() {
            int run_cnt_record = -1;
            while (start_flag_) {
                int run_count;
                RobotBasicState observation;
                {
                    std::lock_guard<std::mutex> lock(observation_mutex_);
                    run_count = state_run_cnt_;
                    if (run_count % policy_ptr_->decimation_ == 0 && run_count != run_cnt_record)
                        observation = rbs_;
                }

                if (run_count % policy_ptr_->decimation_ == 0 && run_count != run_cnt_record) {
                    timespec start_timestamp, end_timestamp;
                    clock_gettime(CLOCK_MONOTONIC, &start_timestamp);
                    auto ra = policy_ptr_->getRobotAction(observation, *(uc_ptr_->GetUserCommand()));
                    
                    MatXf res = ra.ConvertToMat();

                    ri_ptr_->SetJointCommand(res);
                    run_cnt_record = run_count;
                    clock_gettime(CLOCK_MONOTONIC, &end_timestamp);
                    policy_cost_time_ = (end_timestamp.tv_sec - start_timestamp.tv_sec) * 1e3
                                        + (end_timestamp.tv_nsec - start_timestamp.tv_nsec) / 1e6;

                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

    public:
        RLController(const RobotName &robot_name, std::shared_ptr<ControllerData> data_ptr) : MotionController(robot_name, data_ptr) {
            std::memset(&rbs_, 0, sizeof(rbs_));
            if (robot_name_ == RobotName::Lite3) {
                namespace fs = std::filesystem;
                fs::path base = fs::path(__FILE__).parent_path();
                auto model_path = fs::canonical(base / ".." / ".." / "policy" / "policy.onnx");
                lite3_policy_ = std::make_shared<Lite3PolicyRunner>("lite3_policy", model_path.string());
            }

            policy_ptr_ = lite3_policy_;
            if (!policy_ptr_) {
                std::cerr << "error policy" << std::endl;
                exit(0);
            }
            policy_ptr_->DisplayPolicyInfo();
            init_rbs_();
        }

        ~RLController() {}

        virtual void OnEnter() {
            state_run_cnt_ = -1;
            policy_ptr_->OnEnter(rbs_);
            start_flag_ = true;
            run_policy_thread_ = std::thread(std::bind(&RLController::PolicyRunner, this));
        };

        virtual void OnExit() {
            start_flag_ = false;
            if (run_policy_thread_.joinable()) run_policy_thread_.join();
            state_run_cnt_ = -1;
        }

        virtual void Run() {
            std::lock_guard<std::mutex> lock(observation_mutex_);
            UpdateRobotObservation();
            state_run_cnt_++;
        }

        virtual bool LoseControlJudge() {
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
            return PostureUnsafeCheck();
        }

        bool PostureUnsafeCheck() {
            // Vec3f rpy = ri_ptr_->GetImuRpy();
            // if(rpy(0) > 30./180*M_PI || rpy(1) > 45./180*M_PI){
            //     std::cout << "posture value: " << 180./M_PI*rpy.transpose() << std::endl;
            //     return true;
            // }
            return false;
        }

        bool IsReady() override { return policy_ptr_ != nullptr; }
    };
};
