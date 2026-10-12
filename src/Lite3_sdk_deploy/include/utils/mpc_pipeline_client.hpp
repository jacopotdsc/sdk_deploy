#pragma once
#include "mpc_process.hpp"
#include "../../mpc/native/inference_types.hpp"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <atomic>

namespace lite3_mpc {
// Each controller owns its own MPC/WBC pipeline and isolated ROS namespace.
// Owns two child processes; this client only exchanges messages, never performs WBC/MPC.
class MPCPipelineClient {
    using Array=std_msgs::msg::Float64MultiArray;
    using Clock=std::chrono::steady_clock;
    std::shared_ptr<rclcpp::Node> node_;
    MPCProcess mpc_, wbc_;
    std::string namespace_;
    rclcpp::Publisher<Array>::SharedPtr control_pub_;
    rclcpp::Subscription<Array>::SharedPtr output_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mpc_ready_sub_, wbc_ready_sub_;
    mutable std::mutex mutex_;
    bool started_=false, active_=false, mpc_ready_=false, wbc_ready_=false;
    double epoch_=0., state_stamp_=0., plan_stamp_=0.;
    Clock::time_point mpc_received_{}, wbc_received_{}, output_received_{};
    std::optional<WBCOutput> output_;
    static double Parameter(const std::shared_ptr<rclcpp::Node>& node,const std::string& name,double value) {
        if(!node->has_parameter(name)) {
            rcl_interfaces::msg::ParameterDescriptor descriptor; descriptor.read_only=true;
            return node->declare_parameter<double>(name,value,descriptor);
        }
        return node->get_parameter(name).as_double();
    }
    void Publish(bool active,const types::UserCommand& command) {
        Array msg; msg.data={epoch_,active?1.:0.,command.forward_vel_scale,command.side_vel_scale,command.turnning_vel_scale};
        control_pub_->publish(msg);
    }
public:
    explicit MPCPipelineClient(std::shared_ptr<rclcpp::Node> node):node_(std::move(node)) {
        static std::atomic<unsigned> next_id{0};
        namespace_="/mpc_pipeline_"+std::to_string(++next_id);
        control_pub_=node_->create_publisher<Array>(namespace_+"/MPC_CONTROL",1);
        mpc_ready_sub_=node_->create_subscription<std_msgs::msg::Bool>(namespace_+"/MPC_READY",1,[this](const std_msgs::msg::Bool& msg){
            std::lock_guard<std::mutex> lock(mutex_); mpc_ready_=msg.data; mpc_received_=Clock::now();
        });
        wbc_ready_sub_=node_->create_subscription<std_msgs::msg::Bool>(namespace_+"/WBC_READY",1,[this](const std_msgs::msg::Bool& msg){
            std::lock_guard<std::mutex> lock(mutex_); wbc_ready_=msg.data; wbc_received_=Clock::now();
        });
        output_sub_=node_->create_subscription<Array>(namespace_+"/WBC_JOINTS_CMD",1,[this](const Array& msg){
            if(msg.data.size()!=39 || !Finite(msg.data)) return;
            std::lock_guard<std::mutex> lock(mutex_);
            if(!active_ || msg.data[0]!=epoch_) return;
            WBCOutput output;
            for(int i=0;i<12;++i) {
                output.torque(i)=float(msg.data[3+i]);
                output.q_des(i)=float(msg.data[15+i]); output.dq_des(i)=float(msg.data[27+i]);
            }
            if(!output.torque.allFinite() || !output.q_des.allFinite() || !output.dq_des.allFinite()) return;
            output_=std::move(output); state_stamp_=msg.data[1]; plan_stamp_=msg.data[2]; output_received_=Clock::now();
        });
    }
public:
    void Start(const types::Vec3f& limits) {
        if(started_) return;
        const double mpc_hz=Parameter(node_,"mpc_frequency",50.);
        const double wbc_hz=Parameter(node_,"wbc_frequency",50.);
        if(!std::isfinite(mpc_hz) || !std::isfinite(wbc_hz) || mpc_hz<=0 || wbc_hz<=0 || mpc_hz>1000 || wbc_hz>1000)
            throw std::invalid_argument("MPC/WBC frequencies must be in (0,1000] Hz");
        std::vector<std::string> mpc_args{"--ros-args","-r","__ns:="+namespace_,"-p","mpc_frequency:="+std::to_string(mpc_hz)};
        std::vector<std::string> wbc_args{"--ros-args","-r","__ns:="+namespace_,"-p","wbc_frequency:="+std::to_string(wbc_hz),
            "-p","torque_limit_hip:="+std::to_string(limits(0)),
            "-p","torque_limit_thigh:="+std::to_string(limits(1)),
            "-p","torque_limit_knee:="+std::to_string(limits(2))};
        try { wbc_.Start("wbc",wbc_args); mpc_.Start("mpc_srbd",mpc_args); started_=true; }
        catch(...) { mpc_.Stop(); wbc_.Stop(); throw; }
    }
    bool IsReady() const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now=Clock::now();
        return mpc_ready_ && wbc_ready_ && now-mpc_received_<std::chrono::milliseconds(500) &&
            now-wbc_received_<std::chrono::milliseconds(500);
    }
    double Activate(const types::UserCommand& command) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++epoch_; active_=true; output_.reset(); Publish(true,command); return epoch_;
    }
    void Update(double epoch,const types::UserCommand& command) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(active_ && epoch==epoch_) Publish(true,command);
    }
    void Deactivate(double epoch) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(epoch!=epoch_ || !active_) return;
        active_=false; output_.reset(); Publish(false,types::UserCommand{});
    }
    std::optional<WBCOutput> Latest(double epoch,double stamp) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!active_ || epoch!=epoch_ || !output_ || Clock::now()-output_received_>=std::chrono::milliseconds(150) ||
            state_stamp_>stamp+.01 || stamp-state_stamp_>.15 || plan_stamp_>stamp+.01 || stamp-plan_stamp_>.15) return std::nullopt;
        return output_;
    }
    ~MPCPipelineClient() { mpc_.Stop(); wbc_.Stop(); }
};
}
