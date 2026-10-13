#include "mpc_wbc_pipeline_node.hpp"
#include "mpc_runner.hpp"

namespace lite3_mpc {
class MPCNode : public MPCWBCPipelineNode {
    MPCRunner runner_;
    rclcpp::Publisher<Array>::SharedPtr mpc_output_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    double used_epoch_=0., last_stamp_=-std::numeric_limits<double>::infinity();
    bool running_=false;
    double frequency_=50.;
    SolveTimingWindow timing_;
public:
    MPCNode() : MPCWBCPipelineNode("lite3_mpc","MPC_READY"),
        runner_(model_path_,backend_path_,ReadOnlyParameter<double>("mpc_frequency",50.),mpx_root_,
            ReadOnlyParameter<std::string>("mpc_kinematics_backend","mujoco")) {
        const double frequency=get_parameter("mpc_frequency").as_double();
        frequency_=frequency;
        // Read latest sensor values faster than the solve rate; advance MPC in base time,
        // as before the split, including when the GUI slows simulation down.
        const auto period=Period(std::min(1000.,4.*frequency));
        mpc_output_pub_=create_publisher<Array>("MPC_OUTPUT",1);
        RCLCPP_INFO(get_logger(),"Compiling MPC engine at %.1f Hz; kinematics: %s",frequency,
            get_parameter("mpc_kinematics_backend").as_string().c_str());
        const auto compile_start=Clock::now();
        runner_.Compile(); ready_=true;
        RCLCPP_INFO(get_logger(),"MPC engine ready (compiled in %.2f s)",
            std::chrono::duration<double>(Clock::now()-compile_start).count());
        timer_=create_wall_timer(period,[this]{Tick();});
    }
private:
    void OnStateUpdated() override { Tick(); }
    void Tick() {
        if(!ready_) return;
        if(!Fresh()) { if(HardStale()) { running_=false; timing_.Reset(); } return; }
        const bool reset=!running_ || epoch_!=used_epoch_ || state_stamp_<last_stamp_;
        if(!reset && state_stamp_-last_stamp_<1./frequency_-1e-6) return;
        try {
            const RobotState planning_state=state_;
            const auto start=Clock::now();
            const auto mpc_output=runner_.Infer(planning_state,command_,reset);
            const auto line=timing_.Add("MPC",std::chrono::duration<double,std::milli>(Clock::now()-start).count());
            if(!line.empty()) RCLCPP_INFO(get_logger(),"%s",line.c_str());
            Array msg; msg.data={epoch_,state_stamp_};
            msg.data.insert(msg.data.end(),mpc_output.begin(),mpc_output.end());
            mpc_output_pub_->publish(msg);
            used_epoch_=epoch_; last_stamp_=state_stamp_; running_=true;
        } catch(const std::exception& error) {
            ready_=false; RCLCPP_ERROR(get_logger(),"MPC inference failed: %s",error.what());
        }
    }
};
}
int main(int argc,char** argv) {
    rclcpp::init(argc,argv);
    int result=0;
    try {
        pybind11::scoped_interpreter interpreter{};
        // Format Python exceptions while the interpreter and GIL are still alive.
        try { auto node=std::make_shared<lite3_mpc::MPCNode>(); rclcpp::spin(node); }
        catch(const std::exception& error) { std::cerr << "[MPC] " << error.what() << '\n'; result=1; }
    } catch(const std::exception& error) {
        std::cerr << "[MPC] " << error.what() << '\n'; result=1;
    }
    if(rclcpp::ok()) rclcpp::shutdown();
    return result;
}
