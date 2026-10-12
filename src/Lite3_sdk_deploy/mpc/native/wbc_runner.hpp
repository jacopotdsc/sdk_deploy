#pragma once
#include "inference_types.hpp"
#include "joint_mapper.hpp"

namespace lite3_mpc {
// Reusable native C++ WBC engine. Does not invoke Python or solve MPC.
class WBCRunner {
    JointMapper mapper_;
    types::Vec3f limits_;
public:
    explicit WBCRunner(const std::string& model_path,
                          const types::Vec3f& limits=types::Vec3f(40.f,40.f,65.f))
        : mapper_(model_path), limits_(limits) {
        if (!limits_.allFinite() || (limits_.array()<=0).any())
            throw std::invalid_argument("Invalid WBC torque limits");
    }
    WBCOutput Infer(const RobotState& state, const MPCPlan& plan) {
        ValidateState(state);
        if (!Finite(plan)) throw std::invalid_argument("Invalid WBC plan");
        mapper_.Update(state.q.data(),state.dq.data());
        const auto mapped=mapper_.Map(plan);
        if (!Finite(mapped)) throw std::runtime_error("Non-finite WBC output");
        WBCOutput output;
        for(int i=0;i<12;++i) {
            output.torque(i)=std::clamp(float(mapped[i]),-limits_(i%3),limits_(i%3));
            output.q_des(i)=float(mapped[12+i]);
            output.dq_des(i)=float(mapped[24+i]);
        }
        return output;
    }
};
}
