#pragma once
#include "inference_types.hpp"
#include <mujoco/mujoco.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <memory>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <string>

namespace lite3_mpc {
// Reusable numerical engine, with no ROS or scheduling. Caller owns the interpreter
// and calls Compile/Infer/destruction on its GIL-owning thread.
class MPCRunner {
    class MujocoKinematics {
        std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model_{nullptr, mj_deleteModel};
        std::unique_ptr<mjData, decltype(&mj_deleteData)> data_{nullptr, mj_deleteData};
        std::array<int,4> feet_{};
    public:
        explicit MujocoKinematics(const std::string& path) {
            char error[1024]{};
            model_.reset(mj_loadXML(path.c_str(), nullptr, error, sizeof(error)));
            if (!model_) throw std::runtime_error(error);
            if (model_->nq!=19 || model_->nv!=18 || model_->nu!=12)
                throw std::runtime_error("Expected Lite3 nq=19 nv=18 nu=12");
            data_.reset(mj_makeData(model_.get()));
            if (!data_) throw std::runtime_error("MuJoCo data allocation failed");
            const char* names[]={"FL_FOOT_collision","FR_FOOT_collision","HL_FOOT_collision","HR_FOOT_collision"};
            for(int i=0;i<4;++i) {
                feet_[i]=mj_name2id(model_.get(),mjOBJ_GEOM,names[i]);
                if(feet_[i]<0) throw std::runtime_error("Missing foot collision");
            }
        }
        void Prepare(const double* positions,const double* velocities,std::array<double,32>& input,bool contacts_only=false) {
            std::copy_n(positions,19,data_->qpos); std::copy_n(velocities,18,data_->qvel);
            if(contacts_only)mj_fwdPosition(model_.get(),data_.get());
            else mj_forward(model_.get(),data_.get());
            std::copy_n(data_->qpos,7,input.begin()); std::copy_n(data_->qvel,6,input.begin()+7);
            std::fill(input.begin()+25,input.begin()+29,0.);
            for(int leg=0;leg<4;++leg) std::copy_n(data_->geom_xpos+3*feet_[leg],3,input.begin()+13+3*leg);
            for(int c=0;c<data_->ncon;++c) for(int leg=0;leg<4;++leg)
                if(data_->contact[c].dist<=0 && (data_->contact[c].geom1==feet_[leg] || data_->contact[c].geom2==feet_[leg]))
                    input[25+leg]=1.;
        }
    };

    class PinocchioKinematics {
        pinocchio::Model model_;
        std::unique_ptr<pinocchio::Data> data_;
        std::array<pinocchio::FrameIndex,4> feet_{};
        std::array<int,12> q_indices_{};
        // Keep the existing XML collision/contact semantics for both backends.
        MujocoKinematics contacts_;
    public:
        explicit PinocchioKinematics(const std::string& path):contacts_(path) {
            pinocchio::mjcf::buildModel(path,model_,false);
            if(model_.nq!=19 || model_.nv!=18 || model_.joints[1].nq()!=7)
                throw std::runtime_error("Expected floating-base Lite3 model");
            data_=std::make_unique<pinocchio::Data>(model_);
            const char* legs[]={"FL","FR","HL","HR"};
            const char* joints[]={"HipX_joint","HipY_joint","Knee_joint"};
            for(int leg=0;leg<4;++leg) {
                const std::string body=std::string(legs[leg])+"_FOOT";
                if(!model_.existFrame(body,pinocchio::BODY))throw std::runtime_error("Missing foot frame: "+body);
                feet_[leg]=model_.getFrameId(body,pinocchio::BODY);
                for(int j=0;j<3;++j) {
                    const std::string name=std::string(legs[leg])+"_"+joints[j];
                    if(!model_.existJointName(name))throw std::runtime_error("Missing joint: "+name);
                    q_indices_[3*leg+j]=model_.joints[model_.getJointId(name)].idx_q();
                }
            }
        }
        void Prepare(const double* positions,const double* velocities,std::array<double,32>& input) {
            contacts_.Prepare(positions,velocities,input,true);
            Eigen::VectorXd q=Eigen::VectorXd::Zero(model_.nq);
            for(int i=0;i<3;++i)q(i)=positions[i];
            q.segment<4>(3)<<positions[4],positions[5],positions[6],positions[3];
            for(int i=0;i<12;++i)q(q_indices_[i])=positions[7+i];
            pinocchio::forwardKinematics(model_,*data_,q);
            pinocchio::updateFramePlacements(model_,*data_);
            for(int leg=0;leg<4;++leg)
                for(int axis=0;axis<3;++axis)
                    input[13+3*leg+axis]=data_->oMf[feet_[leg]].translation()(axis);
        }
    };

    std::string model_path_, backend_path_,mpx_root_;
    double frequency_;
    std::string kinematics_backend_;
    std::unique_ptr<MujocoKinematics> mujoco_;
    std::unique_ptr<PinocchioKinematics> pinocchio_;
    std::array<double,32> input_{};
    MPCOutput mpc_output_{};
    pybind11::object solver_, step_;
    bool ready_=false;
public:
    MPCRunner(std::string model_path, std::string backend_path, double frequency=50., std::string mpx_root="", std::string kinematics_backend="mujoco")
        : model_path_(std::move(model_path)), backend_path_(std::move(backend_path)), frequency_(frequency),mpx_root_(std::move(mpx_root)),kinematics_backend_(std::move(kinematics_backend)) {
        if(kinematics_backend_!="mujoco" && kinematics_backend_!="pinocchio")
            throw std::invalid_argument("mpc_kinematics_backend must be mujoco or pinocchio");
        if (!std::isfinite(frequency_) || frequency_<=0.) throw std::invalid_argument("Invalid MPC frequency");
    }
    MPCRunner(const MPCRunner&) = delete;
    MPCRunner& operator=(const MPCRunner&) = delete;
    bool IsReady() const { return ready_; }
    void Compile() {
        if (IsReady()) return;
        if(kinematics_backend_=="mujoco")mujoco_=std::make_unique<MujocoKinematics>(model_path_);
        else pinocchio_=std::make_unique<PinocchioKinematics>(model_path_);
        pybind11::module_::import("sys").attr("path").attr("insert")(0,backend_path_);
        auto input = pybind11::memoryview::from_buffer(input_.data(), {32}, {sizeof(double)}, true);
        auto output = pybind11::memoryview::from_buffer(mpc_output_.data(), {40}, {sizeof(double)}, false);
        solver_ = pybind11::module_::import("srbd_solver").attr("Solver")(model_path_,input,output,mpx_root_);
        step_ = solver_.attr("step");
        const auto nominal = solver_.attr("nominal").cast<std::vector<double>>();
        if (nominal.size()!=19) throw std::runtime_error("Invalid nominal MPC state");
        std::array<double,18> zero{};
        PrepareInputs(nominal.data(),zero.data());
        step_(true); step_(false);
        if (!Finite(mpc_output_)) throw std::runtime_error("Invalid MPC warm-up output");
        solver_.attr("finish_warmup")();
        ready_=true;
    }
    MPCOutput Infer(const RobotState& state, const std::array<double,3>& command, bool reset) {
        ValidateState(state);
        if (!Finite(command)) throw std::invalid_argument("Invalid MPC velocity command");
        if (!IsReady()) throw std::runtime_error("MPC is not compiled");
        PrepareInputs(state.q.data(),state.dq.data());
        std::copy(command.begin(),command.end(),input_.begin()+29);
        step_(reset);
        if (!Finite(mpc_output_)) throw std::runtime_error("Non-finite MPC output");
        return mpc_output_;
    }
private:
    void PrepareInputs(const double* positions,const double* velocities) {
        if(mujoco_)mujoco_->Prepare(positions,velocities,input_);
        else if (pinocchio_) pinocchio_->Prepare(positions,velocities,input_);
        else throw std::runtime_error("No kinematics backend available");
    }
};
}
