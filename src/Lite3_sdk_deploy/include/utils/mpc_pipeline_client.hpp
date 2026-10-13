#pragma once
#include "../../mpc/native/inference_types.hpp"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>

namespace lite3_mpc {
// Connects to externally launched MPC/WBC nodes. Owns no processes.
class MPCPipelineClient {
    using Array=std_msgs::msg::Float64MultiArray;
    using Clock=std::chrono::steady_clock;
    std::shared_ptr<rclcpp::Node> node_;
    std::string namespace_;
    rclcpp::Publisher<Array>::SharedPtr control_pub_;
    rclcpp::Subscription<Array>::SharedPtr output_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mpc_ready_sub_, wbc_ready_sub_;
    mutable std::mutex mutex_;
    bool active_=false, mpc_ready_=false, wbc_ready_=false;
    double epoch_=0.;
    Clock::time_point mpc_received_{}, wbc_received_{}, output_received_{};
    std::optional<WBCOutput> output_;
    void Publish(bool active,const types::UserCommand& command) {
        Array msg; msg.data={epoch_,active?1.:0.,command.forward_vel_scale,command.side_vel_scale,command.turnning_vel_scale};
        control_pub_->publish(msg);
    }
public:
    explicit MPCPipelineClient(std::shared_ptr<rclcpp::Node> node):node_(std::move(node)) {
        rcl_interfaces::msg::ParameterDescriptor descriptor; descriptor.read_only=true;
        node_->declare_parameter<std::string>("mpc_namespace","",descriptor);
        namespace_=node_->get_parameter("mpc_namespace").as_string();
        if(!namespace_.empty() && namespace_[0]!='/')
            throw std::invalid_argument("mpc_namespace must be empty or an absolute ROS namespace");
        while(namespace_.size()>1 && namespace_.back()=='/') namespace_.pop_back();
        if(namespace_=="/") namespace_.clear();

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
            output_=std::move(output); output_received_=Clock::now();
        });
    }
public:
    std::string UnavailableReason() const {
        const auto names=node_->get_node_names();
        for(const auto& name:{"lite3_mpc","lite3_wbc"}) {
            const auto full=namespace_+"/"+name;
            if(std::find(names.begin(),names.end(),full)==names.end()) return "Missing ROS node: "+full;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now=Clock::now();
        if(!mpc_ready_ || now-mpc_received_>=std::chrono::milliseconds(500))
            return "MPC is compiling, unavailable or has no recent READY state";
        if(!wbc_ready_ || now-wbc_received_>=std::chrono::milliseconds(500))
            return "WBC is unavailable or has no recent READY state";
        return {};
    }
    bool IsReady() const { return UnavailableReason().empty(); }
    double Activate(const types::UserCommand& command) {
        std::lock_guard<std::mutex> lock(mutex_);
        // External nodes may outlive rl_deploy: avoid reusing its previous session ID.
        const auto now=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        epoch_=std::max(epoch_+1.,double(now));
        active_=true; output_.reset(); Publish(true,command); 
        return epoch_;
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
    std::optional<WBCOutput> Latest(double epoch) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!active_ || epoch!=epoch_ || !output_ || Clock::now()-output_received_>=std::chrono::milliseconds(150)) return std::nullopt;
        return output_;
    }

};
}
