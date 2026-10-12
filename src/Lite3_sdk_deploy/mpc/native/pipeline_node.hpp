#pragma once
#include "inference_types.hpp"
#include "basic_function.hpp"
#include <ament_index_cpp/get_package_prefix.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <drdds/msg/imu_data.hpp>
#include <drdds/msg/joints_data.hpp>
#include <Eigen/Geometry>
#include <chrono>
#include <limits>
#include <iostream>

namespace lite3_mpc {
using Array = std_msgs::msg::Float64MultiArray;
using Clock = std::chrono::steady_clock;
// Shared ROS boundary only; numerical engines do not depend on this class.
// A single-thread executor serializes state callbacks and the node's timer.
class PipelineNode : public rclcpp::Node {
    rclcpp::Subscription<Array>::SharedPtr base_sub_, control_sub_;
    rclcpp::Subscription<drdds::msg::ImuData>::SharedPtr imu_sub_;
    rclcpp::Subscription<drdds::msg::JointsData>::SharedPtr joints_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ready_pub_;
    rclcpp::TimerBase::SharedPtr ready_timer_;
    Clock::time_point base_received_{}, imu_received_{}, joints_received_{}, control_received_{};
    bool have_base_=false, have_imu_=false, have_joints_=false;
    double imu_stamp_=0., joints_stamp_=0.;
protected:
    template<class T> T ReadOnlyParameter(const std::string& name,const T& value) {
        rcl_interfaces::msg::ParameterDescriptor descriptor; descriptor.read_only=true;
        return declare_parameter<T>(name,value,descriptor);
    }
    RobotState state_;
    std::array<double,3> command_{};
    double state_stamp_=0., epoch_=0.;
    bool active_=false, ready_=false;
    std::string model_path_, backend_path_;
    bool HardStale() const {
        const auto now=Clock::now();
        const auto timeout=std::chrono::milliseconds(150);
        return !active_ || !have_base_ || !have_imu_ || !have_joints_ ||
            now-base_received_>=timeout || now-imu_received_>=timeout ||
            now-joints_received_>=timeout || now-control_received_>=timeout;
    }
    bool Fresh() const {
        return !HardStale() && std::abs(state_stamp_-imu_stamp_)<=.01 &&
            std::abs(state_stamp_-joints_stamp_)<=.01;
    }
    static Clock::duration Period(double hz) {
        if (!std::isfinite(hz) || hz<=0 || hz>1000.)
            throw std::invalid_argument("frequency must be in (0,1000] Hz");
        return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1./hz));
    }
public:
    PipelineNode(const std::string& name, const std::string& ready_topic) : Node(name) {
        const auto folder=ament_index_cpp::get_package_prefix("lite3_sdk_deploy")+"/lib/lite3_sdk_deploy";
        model_path_=ReadOnlyParameter<std::string>("model_path",folder+"/Lite3_description/lite3_mjcf/mjcf/Lite3.xml");
        backend_path_=ReadOnlyParameter<std::string>("backend_path",folder+"/mpc");
        ready_pub_=create_publisher<std_msgs::msg::Bool>(ready_topic,1);
        ready_timer_=create_wall_timer(std::chrono::milliseconds(100),[this]{
            std_msgs::msg::Bool msg; msg.data=ready_; ready_pub_->publish(msg);
        });
        base_sub_=create_subscription<Array>("/BASE_DATA",1,[this](const Array& msg){
            if(msg.data.size()!=7 || !Finite(msg.data)) return;
            state_stamp_=msg.data[0];
            std::copy_n(msg.data.begin()+1,3,state_.q.begin());
            std::copy_n(msg.data.begin()+4,3,state_.dq.begin());
            base_received_=Clock::now(); have_base_=true;
        });
        imu_sub_=create_subscription<drdds::msg::ImuData>("/IMU_DATA",rclcpp::SensorDataQoS().keep_last(1),
            [this](const drdds::msg::ImuData& msg){
                const auto& imu=msg.data;
                const std::array<double,6> values{imu.roll,imu.pitch,imu.yaw,imu.omega_x,imu.omega_y,imu.omega_z};
                if(!Finite(values)) return;
                const types::Vec3f rpy(Deg2Rad(imu.roll),Deg2Rad(imu.pitch),Deg2Rad(imu.yaw));
                const Eigen::Quaterniond quat(RpyToRm(rpy).cast<double>());
                state_.q[3]=quat.w(); state_.q[4]=quat.x(); state_.q[5]=quat.y(); state_.q[6]=quat.z();
                std::copy_n(values.begin()+3,3,state_.dq.begin()+3);
                imu_stamp_=rclcpp::Time(msg.header.stamp).seconds(); imu_received_=Clock::now(); have_imu_=true;
            });
        joints_sub_=create_subscription<drdds::msg::JointsData>("/JOINTS_DATA",1,
            [this](const drdds::msg::JointsData& msg){
                for(int i=0;i<12;++i)
                    if(!std::isfinite(msg.data.joints_data[i].position) || !std::isfinite(msg.data.joints_data[i].velocity)) return;
                for(int i=0;i<12;++i) {
                    state_.q[7+i]=msg.data.joints_data[i].position;
                    state_.dq[6+i]=msg.data.joints_data[i].velocity;
                }
                joints_stamp_=rclcpp::Time(msg.header.stamp).seconds(); joints_received_=Clock::now(); have_joints_=true;
            });
        // [session epoch, active, vx, vy, yaw rate]. Controller refreshes this heartbeat.
        control_sub_=create_subscription<Array>("MPC_CONTROL",1,[this](const Array& msg){
            if(msg.data.size()!=5 || !Finite(msg.data) || msg.data[0]<=0 ||
               std::floor(msg.data[0])!=msg.data[0] || (msg.data[1]!=0 && msg.data[1]!=1)) return;
            epoch_=msg.data[0]; active_=msg.data[1]!=0;
            std::copy_n(msg.data.begin()+2,3,command_.begin()); control_received_=Clock::now();
        });
    }
};
}
