#include "joint_mapper.hpp"
#include "basic_function.hpp"
#include <ament_index_cpp/get_package_prefix.hpp>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <drdds/msg/imu_data.hpp>
#include <drdds/msg/joints_data.hpp>
#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <limits>

namespace py=pybind11;
using Clock=std::chrono::steady_clock;
using Array=std_msgs::msg::Float64MultiArray;
class SRBDNode : public rclcpp::Node {
    std::string model_path_, backend_path_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    double timestamp_=0.;
    std::array<double,3> base_position_{};
    Eigen::Quaterniond base_orientation_=Eigen::Quaterniond::Identity();
    std::array<double,3> base_linear_velocity_{};
    std::array<double,3> base_angular_velocity_{};
    std::array<double,12> joint_positions_{};
    std::array<double,12> joint_velocities_{};
    std::array<double,3> velocity_{};
    bool active_=false, have_state_=false, have_imu_=false, have_joints_=false;
    double imu_timestamp_=0., joint_timestamp_=0.;
    Clock::time_point base_received_{}, imu_received_{}, joints_received_{}, active_received_{}, velocity_received_{};
    uint64_t generation_=0;
    std::atomic<bool> ready_{false}, stop_{false};
    rclcpp::Publisher<Array>::SharedPtr command_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ready_pub_;
    rclcpp::Subscription<Array>::SharedPtr base_data_sub_;
    rclcpp::Subscription<drdds::msg::ImuData>::SharedPtr imu_sub_;
    rclcpp::Subscription<drdds::msg::JointsData>::SharedPtr joints_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr velocity_sub_;
    rclcpp::TimerBase::SharedPtr ready_timer_;
    static bool Finite(const double* begin, size_t n) {
        return std::all_of(begin,begin+n,[](double x){return std::isfinite(x);});
    }
    bool Fresh(Clock::time_point now) const {
        const auto timeout=std::chrono::milliseconds(150);
        return active_ && have_state_ && have_imu_ && have_joints_ &&
            now-base_received_<timeout && now-imu_received_<timeout &&
            now-joints_received_<timeout && now-active_received_<timeout &&
            std::abs(timestamp_-imu_timestamp_)<=.01 &&
            std::abs(timestamp_-joint_timestamp_)<=.01;
    }
public:
    SRBDNode() : Node("lite3_mpc_srbd") {
        const auto folder=ament_index_cpp::get_package_prefix("lite3_sdk_deploy")+"/lib/lite3_sdk_deploy";
        model_path_=declare_parameter<std::string>("model_path",folder+"/Lite3_description/lite3_mjcf/mjcf/Lite3.xml");
        backend_path_=folder+"/mpc";
        command_pub_=create_publisher<Array>("/MPC_JOINTS_CMD",1);
        ready_pub_=create_publisher<std_msgs::msg::Bool>("/MPC_READY",1);
        ready_timer_=create_wall_timer(std::chrono::milliseconds(100),[this]{std_msgs::msg::Bool msg;msg.data=ready_;ready_pub_->publish(msg);});
        base_data_sub_=create_subscription<Array>("/BASE_DATA",1,[this](const Array& msg){
            if(msg.data.size()!=7 || !Finite(msg.data.data(),7)) return;
            std::lock_guard<std::mutex> lock(mutex_);
            if(have_state_ && msg.data[0]<timestamp_) ++generation_;
            // /BASE_DATA: [time, x, y, z, vx, vy, vz], in the world frame.
            timestamp_=msg.data[0];
            std::copy_n(msg.data.begin()+1,3,base_position_.begin());
            std::copy_n(msg.data.begin()+4,3,base_linear_velocity_.begin());
            base_received_=Clock::now();have_state_=true;cv_.notify_all();
        });
        imu_sub_=create_subscription<drdds::msg::ImuData>("/IMU_DATA",rclcpp::SensorDataQoS().keep_last(1),
            [this](const drdds::msg::ImuData& msg){
                const auto& imu=msg.data;
                const double values[]={imu.roll,imu.pitch,imu.yaw,imu.omega_x,imu.omega_y,imu.omega_z};
                if(!Finite(values,6))return;
                // Same degrees -> RPY -> rotation matrix conversion as RL.
                const Vec3f rpy(Deg2Rad(imu.roll),Deg2Rad(imu.pitch),Deg2Rad(imu.yaw));
                const Mat3f base_rot_mat=RpyToRm(rpy);
                // MuJoCo qpos needs the same orientation as a [w,x,y,z] quaternion.
                const Eigen::Quaterniond orientation(base_rot_mat.cast<double>());
                std::lock_guard<std::mutex> lock(mutex_);
                base_orientation_=orientation;
                std::copy_n(values+3,3,base_angular_velocity_.begin());
                imu_timestamp_=rclcpp::Time(msg.header.stamp).seconds();
                imu_received_=Clock::now();have_imu_=true;
                cv_.notify_all();
            });
        joints_sub_=create_subscription<drdds::msg::JointsData>("/JOINTS_DATA",1,
            [this](const drdds::msg::JointsData& msg){
                for(int i=0;i<12;++i)
                    if(!std::isfinite(msg.data.joints_data[i].position) ||
                       !std::isfinite(msg.data.joints_data[i].velocity))return;
                std::lock_guard<std::mutex> lock(mutex_);
                // Same first-12 motor order used by Lite3Interface and RL.
                for(int i=0;i<12;++i){
                    joint_positions_[i]=msg.data.joints_data[i].position;
                    joint_velocities_[i]=msg.data.joints_data[i].velocity;
                }
                joint_timestamp_=rclcpp::Time(msg.header.stamp).seconds();
                joints_received_=Clock::now();have_joints_=true;
                cv_.notify_all();
            });
        active_sub_=create_subscription<std_msgs::msg::Bool>("/MPC_ACTIVE",1,[this](const std_msgs::msg::Bool& msg){
            std::lock_guard<std::mutex> lock(mutex_);
            if(active_!=msg.data)++generation_;
            active_=msg.data;active_received_=Clock::now();cv_.notify_all();
        });
        velocity_sub_=create_subscription<geometry_msgs::msg::Twist>("/MPC_CMD_VEL",1,[this](const geometry_msgs::msg::Twist& msg){
            const std::array<double,3> values{msg.linear.x,msg.linear.y,msg.angular.z};
            if(!Finite(values.data(),3))return;
            std::lock_guard<std::mutex> lock(mutex_);velocity_=values;velocity_received_=Clock::now();
        });
    }
    ~SRBDNode() override { Stop(); }
    void Start(){worker_=std::thread([this]{try{Work();}catch(const std::exception& e){ready_=false;RCLCPP_ERROR(get_logger(),"MPC worker failed: %s",e.what());}});}
    void Stop(){stop_=true;cv_.notify_all();if(worker_.joinable())worker_.join();}
private:
    void Work() {
        lite3_mpc::JointMapper joint_mapper(model_path_);
        std::array<double,32> inputs{};
        std::array<double,40> plan{};
        // Python objects are confined to this worker; never touched by ROS callbacks.
        py::gil_scoped_acquire gil;
        py::module_::import("sys").attr("path").attr("insert")(0,backend_path_);
        auto input_view=py::memoryview::from_buffer(inputs.data(),{32},{sizeof(double)},true);
        auto output_view=py::memoryview::from_buffer(plan.data(),{40},{sizeof(double)},false);
        auto solver=py::module_::import("srbd_solver").attr("Solver")(model_path_,input_view,output_view);
        auto step=solver.attr("step");
        const auto nominal=solver.attr("nominal").cast<std::vector<double>>();
        const double mpc_hz=solver.attr("mpc_frequency").cast<double>();
        const double command_hz=solver.attr("whole_body_frequency").cast<double>();
        std::array<double,18> zero_velocity{};
        joint_mapper.Update(nominal.data(),zero_velocity.data());joint_mapper.Inputs(inputs);
        RCLCPP_INFO(get_logger(),"Preparing MPC (C++ ROS/MuJoCo/IK, JAX solver).");
        // Compile reset and continuing paths before advertising readiness.
        step(true);step(false);
        if(!Finite(plan.data(),40))throw std::runtime_error("Invalid MPC warm-up output");
        auto warm_command=joint_mapper.Map(plan);
        if(!Finite(warm_command.data(),36))throw std::runtime_error("Invalid native mapper output");
        solver.attr("finish_warmup")();
        ready_=true;
        RCLCPP_INFO(get_logger(),"MPC READY. Stand with Z / Y, then cycle with C / A.");
        py::gil_scoped_release release;
        bool was_active=false;
        uint64_t used_generation=0;
        double last_sim=-std::numeric_limits<double>::infinity(), last_mpc=last_sim;
        auto next_tick=Clock::now();
        const auto period=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1./command_hz));
        while(!stop_ && rclcpp::ok()) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_until(lock,next_tick,[this]{return stop_.load();});
                if(stop_)break;
            }
            double timestamp;
            std::array<double,19> positions;
            std::array<double,18> velocities;
            uint64_t generation;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if(!Fresh(Clock::now())) {
                    if(active_) {
                        const auto now=Clock::now();
                        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),2000,
                            "Waiting for coherent MPC telemetry: base/imu/joints=%d/%d/%d, stamps=%.6f/%.6f/%.6f, ages(ms)=%.1f/%.1f/%.1f, active=%.1f",
                            have_state_,have_imu_,have_joints_,timestamp_,imu_timestamp_,joint_timestamp_,
                            std::chrono::duration<double,std::milli>(now-base_received_).count(),
                            std::chrono::duration<double,std::milli>(now-imu_received_).count(),
                            std::chrono::duration<double,std::milli>(now-joints_received_).count(),
                            std::chrono::duration<double,std::milli>(now-active_received_).count());
                    }
                    was_active=false;
                    cv_.wait_for(lock,std::chrono::milliseconds(5),[this]{return stop_.load()||Fresh(Clock::now());});
                    next_tick=Clock::now();continue;
                }
                // Assemble the MuJoCo layout only at the mapper boundary:
                // qpos = [base position, quaternion wxyz, joint positions]
                // qvel = [base linear velocity, body angular velocity, joint velocities]
                timestamp=timestamp_;
                std::copy(base_position_.begin(),base_position_.end(),positions.begin());
                positions[3]=base_orientation_.w();positions[4]=base_orientation_.x();
                positions[5]=base_orientation_.y();positions[6]=base_orientation_.z();
                std::copy(joint_positions_.begin(),joint_positions_.end(),positions.begin()+7);
                std::copy(base_linear_velocity_.begin(),base_linear_velocity_.end(),velocities.begin());
                std::copy(base_angular_velocity_.begin(),base_angular_velocity_.end(),velocities.begin()+3);
                std::copy(joint_velocities_.begin(),joint_velocities_.end(),velocities.begin()+6);
                generation=generation_;
                std::copy(velocity_.begin(),velocity_.end(),inputs.begin()+29);
                if(Clock::now()-velocity_received_>=std::chrono::milliseconds(150))
                    std::fill(inputs.begin()+29,inputs.end(),0.);
            }
            next_tick+=period;
            if(next_tick<Clock::now())next_tick=Clock::now()+period;
            if(was_active && generation==used_generation && timestamp==last_sim)continue;
            const bool reset=!was_active || generation!=used_generation || timestamp<last_sim;
            if(reset)last_mpc=-std::numeric_limits<double>::infinity();
            joint_mapper.Update(positions.data(),velocities.data());joint_mapper.Inputs(inputs);
            if(timestamp-last_mpc>=1./mpc_hz-1e-6) {
                {py::gil_scoped_acquire solve_gil;step(reset);}
                last_mpc=timestamp;
                if(!Finite(plan.data(),40))throw std::runtime_error("Non-finite JAX output");
            }
            auto command=joint_mapper.Map(plan);
            if(!Finite(command.data(),36))throw std::runtime_error("Non-finite native joint command");
            {
                // Drop results if deactivated/reactivated/reset while JAX was running.
                std::lock_guard<std::mutex> lock(mutex_);
                if(!Fresh(Clock::now()) || generation_!=generation){was_active=false;continue;}
                Array msg;msg.data.assign(command.begin(),command.end());command_pub_->publish(msg);
            }
            was_active=true;used_generation=generation;last_sim=timestamp;

        }
        ready_=false;
    }
};
int main(int argc,char** argv) {
    py::scoped_interpreter interpreter{};
    rclcpp::init(argc,argv);
    {
        py::gil_scoped_release release;
        auto node=std::make_shared<SRBDNode>();node->Start();rclcpp::spin(node);node->Stop();
    }
    if(rclcpp::ok())rclcpp::shutdown();
}
