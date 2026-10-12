#pragma once
#include "inference_types.hpp"
#include "joint_mapper.hpp"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <memory>

namespace lite3_mpc {
// Reusable numerical engine, with no ROS or scheduling. Caller owns the interpreter
// and calls Compile/Infer/destruction on its GIL-owning thread.
class MPCRunner {
    std::string model_path_, backend_path_;
    double frequency_;
    std::unique_ptr<JointMapper> kinematics_;
    std::array<double,32> input_{};
    MPCPlan plan_{};
    pybind11::object solver_, step_;
    bool ready_=false;
public:
    MPCRunner(std::string model_path, std::string backend_path, double frequency=50.)
        : model_path_(std::move(model_path)), backend_path_(std::move(backend_path)), frequency_(frequency) {
        if (!std::isfinite(frequency_) || frequency_<=0.) throw std::invalid_argument("Invalid MPC frequency");
    }
    MPCRunner(const MPCRunner&) = delete;
    MPCRunner& operator=(const MPCRunner&) = delete;
    bool IsReady() const { return ready_; }
    void Compile() {
        if (IsReady()) return;
        kinematics_ = std::make_unique<JointMapper>(model_path_);
        pybind11::module_::import("sys").attr("path").attr("insert")(0,backend_path_);
        auto input = pybind11::memoryview::from_buffer(input_.data(), {32}, {sizeof(double)}, true);
        auto output = pybind11::memoryview::from_buffer(plan_.data(), {40}, {sizeof(double)}, false);
        solver_ = pybind11::module_::import("srbd_solver").attr("Solver")(model_path_,input,output,frequency_);
        step_ = solver_.attr("step");
        const auto nominal = solver_.attr("nominal").cast<std::vector<double>>();
        if (nominal.size()!=19) throw std::runtime_error("Invalid nominal MPC state");
        std::array<double,18> zero{};
        kinematics_->Update(nominal.data(),zero.data()); kinematics_->Inputs(input_);
        step_(true); step_(false);
        if (!Finite(plan_)) throw std::runtime_error("Invalid MPC warm-up plan");
        solver_.attr("finish_warmup")();
        ready_=true;
    }
    MPCPlan Infer(const RobotState& state, const std::array<double,3>& command, bool reset) {
        ValidateState(state);
        if (!Finite(command)) throw std::invalid_argument("Invalid MPC velocity command");
        if (!IsReady()) throw std::runtime_error("MPC is not compiled");
        kinematics_->Update(state.q.data(),state.dq.data()); kinematics_->Inputs(input_);
        std::copy(command.begin(),command.end(),input_.begin()+29);
        step_(reset);
        if (!Finite(plan_)) throw std::runtime_error("Non-finite MPC plan");
        return plan_;
    }
};
}
