#include "pipeline_node.hpp"
#include "wbc_runner.hpp"

namespace lite3_mpc {
class WBCNode : public PipelineNode {
    WBCRunner runner_;
    MPCPlan plan_{};
    double plan_epoch_=0., plan_stamp_=0.;
    Clock::time_point plan_received_{};
    rclcpp::Subscription<Array>::SharedPtr plan_sub_;
    rclcpp::Publisher<Array>::SharedPtr command_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
public:
    WBCNode() : PipelineNode("lite3_wbc","WBC_READY"),
        runner_(model_path_,types::Vec3f(
            ReadOnlyParameter<double>("torque_limit_hip",40.),
            ReadOnlyParameter<double>("torque_limit_thigh",40.),
            ReadOnlyParameter<double>("torque_limit_knee",65.))) {
        const double frequency=declare_parameter<double>("wbc_frequency",50.);
        const auto period=Period(frequency);
        command_pub_=create_publisher<Array>("WBC_JOINTS_CMD",1);
        plan_sub_=create_subscription<Array>("MPC_PLAN",1,[this](const Array& msg){
            if(msg.data.size()!=42 || !Finite(msg.data) || msg.data[0]!=epoch_ || !active_) return;
            plan_epoch_=msg.data[0]; plan_stamp_=msg.data[1];
            std::copy_n(msg.data.begin()+2,40,plan_.begin()); plan_received_=Clock::now();
        });
        ready_=true;
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
                timer_->cancel(); timer_=create_wall_timer(Period(frequency),[this]{Tick();});
                RCLCPP_INFO(get_logger(),"WBC frequency changed to %.1f Hz",frequency);
            }
            return result;
        });
        RCLCPP_INFO(get_logger(),"Native WBC ready at %.1f Hz; consumes latest MPC plan",frequency);
    }
private:
    void Tick() {
        // Do not wait for MPC: reuse a completed plan until its freshness expires.
        if(!ready_ || !Fresh() || plan_epoch_!=epoch_ ||
           Clock::now()-plan_received_>=std::chrono::milliseconds(150) ||
           plan_stamp_>state_stamp_+.01 || state_stamp_-plan_stamp_>.15) return;
        try {
            const auto output=runner_.Infer(state_,plan_);
            Array msg; msg.data={epoch_,state_stamp_,plan_stamp_};
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
    try { auto node=std::make_shared<lite3_mpc::WBCNode>(); rclcpp::spin(node); }
    catch(const std::exception& error) { std::cerr << "[WBC] " << error.what() << '\n'; result=1; }
    if(rclcpp::ok()) rclcpp::shutdown();
    return result;
}
