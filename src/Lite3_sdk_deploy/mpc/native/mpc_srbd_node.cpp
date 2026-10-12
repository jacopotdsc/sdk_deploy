#include "pipeline_node.hpp"
#include "mpc_runner.hpp"

namespace lite3_mpc {
class MPCNode : public PipelineNode {
    MPCRunner runner_;
    rclcpp::Publisher<Array>::SharedPtr plan_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    double used_epoch_=0., last_stamp_=-std::numeric_limits<double>::infinity();
    bool running_=false;
public:
    MPCNode() : PipelineNode("lite3_mpc","MPC_READY"),
        runner_(model_path_,backend_path_,ReadOnlyParameter<double>("mpc_frequency",50.)) {
        const double frequency=get_parameter("mpc_frequency").as_double();
        const auto period=Period(frequency);
        plan_pub_=create_publisher<Array>("MPC_PLAN",1);
        RCLCPP_INFO(get_logger(),"Compiling MPC engine at %.1f Hz",frequency);
        runner_.Compile(); ready_=true;
        timer_=create_wall_timer(period,[this]{Tick();});
    }
private:
    void Tick() {
        if(!ready_) return;
        if(!Fresh()) { if(HardStale()) running_=false; return; }
        const bool reset=!running_ || epoch_!=used_epoch_ || state_stamp_<last_stamp_;
        if(!reset && state_stamp_==last_stamp_) return;
        try {
            const auto plan=runner_.Infer(state_,command_,reset);
            Array msg; msg.data={epoch_,state_stamp_};
            msg.data.insert(msg.data.end(),plan.begin(),plan.end()); plan_pub_->publish(msg);
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
