#include "joint_mapper.hpp"
#include <ament_index_cpp/get_package_prefix.hpp>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
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
    std::array<double,38> observation_{};
    std::array<double,3> velocity_{};
    Clock::time_point state_received_{}, active_received_{}, velocity_received_{};
    bool active_=false, have_state_=false;
    uint64_t generation_=0;
    std::atomic<bool> ready_{false}, stop_{false};
    double startup_delay_=0.;
    rclcpp::Publisher<Array>::SharedPtr command_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ready_pub_;
    rclcpp::Subscription<Array>::SharedPtr state_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr velocity_sub_;
    rclcpp::TimerBase::SharedPtr ready_timer_;
    bool Fresh(Clock::time_point now) const {
        return active_ && have_state_ && now-active_received_<std::chrono::milliseconds(150) &&
               now-state_received_<std::chrono::milliseconds(150);
    }
    static bool Finite(const double* begin, size_t n) {
        return std::all_of(begin,begin+n,[](double x){return std::isfinite(x);});
    }
public:
    SRBDNode() : Node("lite3_mpc_srbd") {
        const auto folder=ament_index_cpp::get_package_prefix("lite3_sdk_deploy")+"/lib/lite3_sdk_deploy";
        model_path_=declare_parameter<std::string>("model_path",folder+"/Lite3_description/lite3_mjcf/mjcf/Lite3.xml");
        backend_path_=folder+"/mpc";
        // Test-only delay makes the readiness gate deterministic, including with a JAX cache.
        const char* delay=std::getenv("LITE3_MPC_STARTUP_DELAY");
        startup_delay_=declare_parameter<double>("startup_delay",delay?std::stod(delay):0.);
        if(!std::isfinite(startup_delay_) || startup_delay_<0.) throw std::runtime_error("Invalid startup_delay");
        command_pub_=create_publisher<Array>("lite3/mpc/torque",1);
        ready_pub_=create_publisher<std_msgs::msg::Bool>("lite3/mpc/ready",1);
        ready_timer_=create_wall_timer(std::chrono::milliseconds(100),[this]{std_msgs::msg::Bool msg;msg.data=ready_;ready_pub_->publish(msg);});
        state_sub_=create_subscription<Array>("lite3/mpc/sim_state",1,[this](const Array& msg){
            if(msg.data.size()!=38 || !Finite(msg.data.data(),38)) return;
            double norm=0.;for(int i=4;i<8;++i)norm+=msg.data[i]*msg.data[i];if(norm<.25)return;
            std::lock_guard<std::mutex> lock(mutex_);
            if(have_state_ && msg.data[0]<observation_[0]) ++generation_;
            std::copy_n(msg.data.begin(),38,observation_.begin());
            state_received_=Clock::now();have_state_=true;cv_.notify_all();
        });
        active_sub_=create_subscription<std_msgs::msg::Bool>("lite3/mpc/active",1,[this](const std_msgs::msg::Bool& msg){
            std::lock_guard<std::mutex> lock(mutex_);
            if(active_!=msg.data)++generation_;
            active_=msg.data;active_received_=Clock::now();cv_.notify_all();
        });
        velocity_sub_=create_subscription<geometry_msgs::msg::Twist>("lite3/mpc/cmd_vel",1,[this](const geometry_msgs::msg::Twist& msg){
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
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock,std::chrono::duration<double>(startup_delay_),[this]{return stop_.load();});
            if(stop_)return;
        }
        lite3_mpc::JointMapper mapper(model_path_);
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
        mapper.Update(nominal.data(),zero_velocity.data());mapper.Inputs(inputs);
        RCLCPP_INFO(get_logger(),"Preparing MPC (C++ ROS/MuJoCo/IK, JAX solver).");
        // Compile reset and continuing paths before advertising readiness.
        step(true);step(false);
        if(!Finite(plan.data(),40))throw std::runtime_error("Invalid MPC warm-up output");
        auto warm_command=mapper.Map(plan);
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
        auto log_start=Clock::now();size_t solves=0,outputs=0;double solve_ms=0.,native_ms=0.;
        while(!stop_ && rclcpp::ok()) {
            std::array<double,38> state;
            uint64_t generation;
            Clock::time_point snapshot_received;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_until(lock,next_tick,[this]{return stop_.load();});
                if(stop_)break;
                const auto now=Clock::now();
                if(!Fresh(now)) {
                    was_active=false;
                    cv_.wait_for(lock,std::chrono::milliseconds(5),[this]{return stop_.load()||Fresh(Clock::now());});
                    next_tick=Clock::now();continue;
                }
                state=observation_;generation=generation_;snapshot_received=state_received_;
                std::copy(velocity_.begin(),velocity_.end(),inputs.begin()+29);
                if(now-velocity_received_>=std::chrono::milliseconds(150))std::fill(inputs.begin()+29,inputs.end(),0.);
            }
            next_tick+=period;
            if(next_tick<Clock::now())next_tick=Clock::now()+period;
            if(was_active && generation==used_generation && state[0]==last_sim)continue;
            const bool reset=!was_active || generation!=used_generation || state[0]<last_sim;
            if(reset)last_mpc=-std::numeric_limits<double>::infinity();
            auto native_start=Clock::now();
            mapper.Update(state.data()+1,state.data()+20);mapper.Inputs(inputs);
            native_ms+=std::chrono::duration<double,std::milli>(Clock::now()-native_start).count();
            if(state[0]-last_mpc>=1./mpc_hz-1e-6) {
                auto solve_start=Clock::now();
                {py::gil_scoped_acquire solve_gil;step(reset);}
                solve_ms+=std::chrono::duration<double,std::milli>(Clock::now()-solve_start).count();
                ++solves;last_mpc=state[0];
                if(!Finite(plan.data(),40))throw std::runtime_error("Non-finite JAX output");
            }
            native_start=Clock::now();auto command=mapper.Map(plan);
            native_ms+=std::chrono::duration<double,std::milli>(Clock::now()-native_start).count();
            if(!Finite(command.data(),36))throw std::runtime_error("Non-finite native joint command");
            {
                // Drop results if deactivated/reactivated/reset while JAX was running.
                std::lock_guard<std::mutex> lock(mutex_);
                if(!Fresh(Clock::now()) || generation_!=generation ||
                   Clock::now()-snapshot_received>=std::chrono::milliseconds(150)){was_active=false;continue;}
                Array msg;msg.data.assign(command.begin(),command.end());command_pub_->publish(msg);++outputs;
            }
            was_active=true;used_generation=generation;last_sim=state[0];
            const double seconds=std::chrono::duration<double>(Clock::now()-log_start).count();
            if(seconds>=2.) {
                RCLCPP_INFO(get_logger(),"MPC %.1f Hz, output %.1f Hz; JAX call avg %.3f ms; native avg %.3f ms/output",solves/seconds,outputs/seconds,solves?solve_ms/solves:0.,outputs?native_ms/outputs:0.);
                log_start=Clock::now();solves=outputs=0;solve_ms=native_ms=0.;
            }
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
