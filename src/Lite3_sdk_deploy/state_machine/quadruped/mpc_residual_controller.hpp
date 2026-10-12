#pragma once
#include "mpx_controller.hpp"
#include "lite3_policy_runner.hpp"
#include <filesystem>
#include <optional>

namespace q {
class MPCResidualController : public MPXController {
    std::unique_ptr<Lite3PolicyRunner> policy_;
    std::optional<RobotAction> network_action_;
    double network_stamp_ = -std::numeric_limits<double>::infinity();
    static constexpr double kNetworkPeriod = .02;
    MatXf combine_mpc_network(const lite3_mpc::WBCOutput& mpc_output, const RobotAction& network_output) const {
        // Build the final Lite3 command here using either inference output.
        // Network fields: tau_ff, goal_joint_pos, goal_joint_vel, kp, kd.
        // Torque and joint references come from the same MPC plan and native mapper.
        RobotAction command;
        VecXf delta_q = network_output.goal_joint_pos;
        delta_q.setZero(); // Disable the network correction until the residual policy is ready.
        command.tau_ff = mpc_output.torque;
        command.goal_joint_pos = mpc_output.q_des + delta_q;
        command.goal_joint_vel = mpc_output.dq_des;
        command.kp = VecXf::Constant(12, kJointKp);
        command.kd = VecXf::Constant(12, kJointKd);

        // Edit any of the five fields above using network_output before returning.
        return command.ConvertToMat(); // Columns: [kp, q_des, kd, dq_des, torque].
    }
protected:
    void ResetPolicy() override {
        policy_->OnEnter(RobotBasicState{});
        network_action_.reset();
        network_stamp_ = -std::numeric_limits<double>::infinity();
    }
    MatXf ComputeCommand(const lite3_mpc::WBCOutput& mpc,
                        const RobotBasicState& snapshot, const UserCommand& command,
                        double state_stamp) override {
        // MPC and WBC run independently in their ROS processes.
        // Run a due ONNX inference locally and combine with the latest WBC command.
        if (state_stamp < network_stamp_) ResetPolicy();
        if (!network_action_ || state_stamp - network_stamp_ >= kNetworkPeriod - 1e-6) {
            auto action = policy_->getRobotAction(snapshot,command);
            for(const auto* field:{&action.tau_ff,&action.goal_joint_pos,&action.goal_joint_vel,&action.kp,&action.kd})
                if(field->size()!=12 || !field->allFinite())throw std::runtime_error("Invalid residual policy output");
            network_action_ = std::move(action);
            network_stamp_ = state_stamp;
        }
        return combine_mpc_network(mpc,*network_action_);
    }
public:
    MPCResidualController(RobotName robot,std::shared_ptr<ControllerData> data)
        : MPXController(robot,data) {
        const auto model=std::filesystem::canonical(std::filesystem::path(__FILE__).parent_path()/".."/".."/"policy"/"residual.onnx");
        policy_=std::make_unique<Lite3PolicyRunner>("residual_policy",model.string(),kJointKp,kJointKd);
    }
};
}
