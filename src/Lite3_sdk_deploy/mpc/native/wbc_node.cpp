#include "mpc_wbc_pipeline_node.hpp"
#include "wbc_runner.hpp"

namespace lite3_mpc {
class WBCNode : public MPCWBCPipelineNode {
    WBCRunner runner_;
    MPCOutput mpc_output_{};
    double frequency_=100.;
    double mpc_output_epoch_=0., mpc_output_stamp_=0.;
    Clock::time_point mpc_output_received_{};
    SolveTimingWindow timing_;
    rclcpp::Subscription<Array>::SharedPtr mpc_output_sub_;
    rclcpp::Publisher<Array>::SharedPtr command_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
public:
    WBCNode() : MPCWBCPipelineNode("lite3_wbc","WBC_READY"),
        runner_(model_path_,backend_path_,types::Vec3f(
            ReadOnlyParameter<double>("torque_limit_hip",40.),
            ReadOnlyParameter<double>("torque_limit_thigh",40.),
            ReadOnlyParameter<double>("torque_limit_knee",65.)),
            ReadOnlyParameter<std::string>("wbc_backend","ik"),mpx_root_,
            ReadOnlyParameter<std::string>("wbc_targets_backend","integration")) {
        const double frequency=declare_parameter<double>("wbc_frequency",100.);
        const auto period=Period(frequency);
        frequency_=frequency;
        command_pub_=create_publisher<Array>("WBC_JOINTS_CMD",1);
        mpc_output_sub_=create_subscription<Array>("MPC_OUTPUT",1,[this](const Array& msg){
            if(msg.data.size()!=42 || !Finite(msg.data) || msg.data[0]!=epoch_ || !active_) return;
            mpc_output_epoch_=msg.data[0]; mpc_output_stamp_=msg.data[1];
            std::copy_n(msg.data.begin()+2,40,mpc_output_.begin()); mpc_output_received_=Clock::now();

        });
        RCLCPP_INFO(get_logger(),"Preparing WBC backend: %s",get_parameter("wbc_backend").as_string().c_str());
        const auto compile_start=Clock::now();
        runner_.Compile();
        ready_=true;
        RCLCPP_INFO(get_logger(),"WBC engine ready (compiled in %.2f s)",
            std::chrono::duration<double>(Clock::now()-compile_start).count());
        timer_=create_wall_timer(period,[this]{Tick();});
        parameter_callback_=add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& parameters) {
            rcl_interfaces::msg::SetParametersResult result; result.successful=true;
            // Validate the complete transaction before changing the timer.
            double frequency=0.;
            try {
                for(const auto& parameter:parameters) if(parameter.get_name()=="wbc_frequency") {
                    frequency=parameter.as_double(); Period(frequency);
                }
            } catch(const std::exception& error) {
                result.successful=false; result.reason=error.what(); return result;
            }
            if(frequency>0.) {
                frequency_=frequency;
                timer_->cancel(); timer_=create_wall_timer(Period(frequency),[this]{Tick();});
                RCLCPP_INFO(get_logger(),"WBC frequency changed to %.1f Hz",frequency);
            }
            return result;
        });
        RCLCPP_INFO(get_logger(),"WBC backend %s ready at %.1f Hz; publishes joint targets",
            get_parameter("wbc_backend").as_string().c_str(),frequency);
    }
private:
    void Tick() {
        // Use the latest MPC output and latest value of each sensor independently.
        if(!ready_ || !Fresh() || mpc_output_epoch_!=epoch_ ||
           Clock::now()-mpc_output_received_>=std::chrono::milliseconds(150)) { timing_.Reset(); return; }
        try {
            const auto start=Clock::now();
            const auto calculation=runner_.Infer(state_,mpc_output_);
            const double dt=1./frequency_;
            WBCOutput output; output.torque=calculation.torque;
            if(calculation.joint_targets) {
                output.q_des=calculation.joint_targets->q_des;
                output.dq_des=calculation.joint_targets->dq_des;
            } else for(int i=0;i<12;++i) {
                const double qacc=calculation.qacc[6+i];
                const auto limits=runner_.JointPositionLimits(i);
                const double q=state_.q[7+i]+state_.dq[6+i]*dt+.5*qacc*dt*dt;
                const double dq=state_.dq[6+i]+qacc*dt;
                output.q_des(i)=float(std::clamp(q,limits.first,limits.second));
                output.dq_des(i)=float(std::clamp(dq,-10.,10.));
            }
            if(!output.q_des.allFinite() || !output.dq_des.allFinite())
                throw std::runtime_error("Non-finite integrated WBC targets");
            const auto line=timing_.Add("WBC",std::chrono::duration<double,std::milli>(Clock::now()-start).count());
            if(!line.empty()) RCLCPP_INFO(get_logger(),"%s",line.c_str());
            Array msg; msg.data={epoch_,state_stamp_,mpc_output_stamp_};
            for(const auto* values:{&output.torque,&output.q_des,&output.dq_des})
                msg.data.insert(msg.data.end(),values->data(),values->data()+values->size());
            command_pub_->publish(msg);
        } catch(const std::exception& error) {
            ready_=false; RCLCPP_ERROR(get_logger(),"WBC inference failed: %s",error.what());
        }
    }
};
}
int main(int argc,char** argv) {
    rclcpp::init(argc,argv);
    int result=0;
    try {
        pybind11::scoped_interpreter interpreter{};
        try { auto node=std::make_shared<lite3_mpc::WBCNode>(); rclcpp::spin(node); }
        catch(const std::exception& error) { std::cerr << "[WBC] " << error.what() << '\n'; result=1; }
    }
    catch(const std::exception& error) { std::cerr << "[WBC] " << error.what() << '\n'; result=1; }
    if(rclcpp::ok()) rclcpp::shutdown();
    return result;
}
