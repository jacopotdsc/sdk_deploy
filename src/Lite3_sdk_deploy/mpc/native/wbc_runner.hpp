#pragma once
#include "inference_types.hpp"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <string>
#include <mujoco/mujoco.h>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <Eigen/SVD>
#include <memory>

namespace lite3_mpc {
// Owns only the backend selected at startup. Caller owns the interpreter/GIL.
// Both engines return torque/qacc; target integration stays in the ROS node.
class WBCRunner {
    class IKWBC {
        std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model_{nullptr, mj_deleteModel};
        std::unique_ptr<mjData, decltype(&mj_deleteData)> data_{nullptr, mj_deleteData};
        std::unique_ptr<mjData, decltype(&mj_deleteData)> ik_{nullptr, mj_deleteData};
        std::array<int,4> feet_{};
        std::array<double,54> jac_{}, rotation_{};
        using Jacobian = Eigen::Matrix<double,3,18,Eigen::RowMajor>;
    public:
        explicit IKWBC(const std::string& path) {
            char error[1024]{};
            model_.reset(mj_loadXML(path.c_str(), nullptr, error, sizeof(error)));
            if (!model_) throw std::runtime_error(error);
            if (model_->nq!=19 || model_->nv!=18 || model_->nu!=12)
                throw std::runtime_error("Expected Lite3 nq=19 nv=18 nu=12");
            data_.reset(mj_makeData(model_.get())); ik_.reset(mj_makeData(model_.get()));
            if (!data_ || !ik_) throw std::runtime_error("MuJoCo data allocation failed");
            const char* names[]={"FL_FOOT_collision","FR_FOOT_collision","HL_FOOT_collision","HR_FOOT_collision"};
            for(int i=0;i<4;++i) {
                feet_[i]=mj_name2id(model_.get(),mjOBJ_GEOM,names[i]);
                if(feet_[i]<0) throw std::runtime_error("Missing foot collision");
            }
        }
        void Update(const double* positions, const double* velocities) {
            std::copy_n(positions,19,data_->qpos); std::copy_n(velocities,18,data_->qvel);
            mj_forward(model_.get(),data_.get());
        }
        // Solver output: foot positions/velocities, GRF, planned contacts (40 doubles).
        std::array<double,36> Map(const std::array<double,40>& mpc_output) {
            std::array<double,36> out{};
            std::copy_n(data_->qfrc_bias+6,12,out.begin());
            std::copy_n(data_->qpos+7,12,out.begin()+12);
            std::copy_n(data_->qpos,19,ik_->qpos);
            Eigen::Map<Jacobian> jac(jac_.data());
            const Eigen::Map<const Eigen::Matrix<double,6,1>> base_velocity(data_->qvel);
            for(int leg=0;leg<4;++leg) {
                const int geom=feet_[leg], body=model_->geom_bodyid[geom], dof=6+3*leg, joint=7+3*leg;
                mj_jac(model_.get(),data_.get(),jac_.data(),rotation_.data(),data_->geom_xpos+3*geom,body);
                Eigen::Matrix3d local=jac.block<3,3>(0,dof);
                const Eigen::Map<const Eigen::Vector3d> force(mpc_output.data()+24+3*leg);
                Eigen::Map<Eigen::Vector3d>(out.data()+3*leg) -= local.transpose()*(force*mpc_output[36+leg]);
                const Eigen::Map<const Eigen::Vector3d> ref_velocity(mpc_output.data()+12+3*leg);
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(local,Eigen::ComputeFullU|Eigen::ComputeFullV);
                svd.setThreshold(1e-3);
                Eigen::Map<Eigen::Vector3d>(out.data()+24+3*leg)=svd.solve(ref_velocity-jac.block<3,6>(0,0)*base_velocity);
                for(int iteration=0;iteration<6;++iteration) {
                    mj_forward(model_.get(),ik_.get());
                    mj_jac(model_.get(),ik_.get(),jac_.data(),rotation_.data(),ik_->geom_xpos+3*geom,body);
                    Eigen::Vector3d error=Eigen::Map<const Eigen::Vector3d>(mpc_output.data()+3*leg)-Eigen::Map<const Eigen::Vector3d>(ik_->geom_xpos+3*geom);
                    if(error.norm()<1e-4) break;
                    local=jac.block<3,3>(0,dof);
                    Eigen::Vector3d delta=local.transpose()*(local*local.transpose()+1e-5*Eigen::Matrix3d::Identity()).ldlt().solve(error);
                    for(int j=0;j<3;++j) {
                        ik_->qpos[joint+j]+=std::clamp(delta[j],-.15,.15);
                        ik_->qpos[joint+j]=std::clamp(ik_->qpos[joint+j],model_->jnt_range[2*(1+3*leg+j)],model_->jnt_range[2*(1+3*leg+j)+1]);
                    }
                }
                std::copy_n(ik_->qpos+joint,3,out.begin()+12+3*leg);
            }
            for(int i=0;i<12;++i) {
                out[i]=std::clamp(out[i],model_->actuator_ctrlrange[2*i],model_->actuator_ctrlrange[2*i+1]);
                out[24+i]=std::clamp(out[24+i],-10.,10.);
            }
            return out;
        }
        WBCCalculation Infer(const RobotState& state,const MPCOutput& mpc_output) {
            Update(state.q.data(),state.dq.data());
            const auto mapped=Map(mpc_output);
            WBCCalculation output;output.joint_targets.emplace();
            for(int i=0;i<12;++i) {
                output.torque(i)=float(mapped[i]);
                output.joint_targets->q_des(i)=float(mapped[12+i]);
                output.joint_targets->dq_des(i)=float(mapped[24+i]);
            }
            return output;
        }
        std::vector<std::array<double,2>> Limits(bool torque) const {
            std::vector<std::array<double,2>> result(12);
            for(int i=0;i<12;++i) {
                const auto* values=torque?model_->actuator_ctrlrange:model_->jnt_range+2;
                result[i]={values[2*i],values[2*i+1]};
            }
            return result;
        }
    };

    class JaxWBC {
        std::array<double,77> input_{};
        std::array<double,30> result_{};
        pybind11::object solver_,step_;
    public:
        JaxWBC(const std::string& model_path,const std::string& mpx_root) {
            auto input=pybind11::memoryview::from_buffer(input_.data(),{77},{sizeof(double)},true);
            auto output=pybind11::memoryview::from_buffer(result_.data(),{30},{sizeof(double)},false);
            solver_=pybind11::module_::import("wbc_solver").attr("Solver")(model_path,input,output,mpx_root);
            step_=solver_.attr("step");
            const auto nominal=solver_.attr("nominal").cast<std::vector<double>>();
            if(nominal.size()!=19)throw std::runtime_error("Invalid WBC model dimensions");
            std::copy(nominal.begin(),nominal.end(),input_.begin());
            step_();step_();
            if(!Finite(result_))throw std::runtime_error("Invalid MPX WBC warm-up output");
        }
        std::vector<std::array<double,2>> Limits(const char* name) const {
            return solver_.attr(name).cast<std::vector<std::array<double,2>>>();
        }
        WBCCalculation Infer(const RobotState& state,const MPCOutput& mpc_output) {
            std::copy(state.q.begin(),state.q.end(),input_.begin());
            std::copy(state.dq.begin(),state.dq.end(),input_.begin()+19);
            std::copy(mpc_output.begin(),mpc_output.end(),input_.begin()+37);
            step_();
            if(!Finite(result_))throw std::runtime_error("Non-finite MPX WBC output");
            WBCCalculation output;
            for(int i=0;i<12;++i)output.torque(i)=float(result_[i]);
            std::copy_n(result_.begin()+12,18,output.qacc.begin());
            return output;
        }
    };

    class PinocchioWBC {
        pinocchio::Model model_;
        std::unique_ptr<pinocchio::Data> data_;
        std::array<pinocchio::FrameIndex,4> feet_;
        std::array<int,12> q_indices_,v_indices_;
        Eigen::VectorXd Configuration(const RobotState& state) const {
            Eigen::VectorXd q=Eigen::VectorXd::Zero(model_.nq);
            for(int i=0;i<3;++i)q(i)=state.q[i];
            q.segment<4>(3)<<state.q[4],state.q[5],state.q[6],state.q[3];
            for(int i=0;i<12;++i)q(q_indices_[i])=state.q[7+i];
            return q;
        }
        Eigen::VectorXd Velocity(const RobotState& state) const {
            Eigen::VectorXd v=Eigen::VectorXd::Zero(model_.nv);
            const Eigen::Quaterniond quat(state.q[3],state.q[4],state.q[5],state.q[6]);
            v.head<3>()=quat.toRotationMatrix().transpose()*Eigen::Map<const Eigen::Vector3d>(state.dq.data());
            v.segment<3>(3)=Eigen::Map<const Eigen::Vector3d>(state.dq.data()+3);
            for(int i=0;i<12;++i)v(v_indices_[i])=state.dq[6+i];
            return v;
        }
        void Update(const Eigen::VectorXd& q) {
            pinocchio::computeJointJacobians(model_,*data_,q);
            pinocchio::updateFramePlacements(model_,*data_);
        }
        Eigen::MatrixXd Jacobian(int leg) {
            Eigen::MatrixXd full=Eigen::MatrixXd::Zero(6,model_.nv);
            pinocchio::getFrameJacobian(model_,*data_,feet_[leg],pinocchio::LOCAL_WORLD_ALIGNED,full);
            return full.topRows(3);
        }
    public:
        explicit PinocchioWBC(const std::string& path) {
            pinocchio::mjcf::buildModel(path,model_,false);
            if(model_.nq!=19 || model_.nv!=18 || model_.joints[1].nq()!=7)
                throw std::runtime_error("Expected floating-base Lite3 model");
            data_=std::make_unique<pinocchio::Data>(model_);
            const char* legs[]={"FL","FR","HL","HR"};
            const char* joints[]={"HipX_joint","HipY_joint","Knee_joint"};
            for(int leg=0;leg<4;++leg) {
                // The SDK XML's FOOT_collision sphere is centered on this body origin.
                const std::string body=std::string(legs[leg])+"_FOOT";
                if(!model_.existFrame(body,pinocchio::BODY))throw std::runtime_error("Missing foot frame: "+body);
                feet_[leg]=model_.getFrameId(body,pinocchio::BODY);
                for(int j=0;j<3;++j) {
                    const std::string name=std::string(legs[leg])+"_"+joints[j];
                    if(!model_.existJointName(name))throw std::runtime_error("Missing joint: "+name);
                    const auto id=model_.getJointId(name);
                    q_indices_[3*leg+j]=model_.joints[id].idx_q();
                    v_indices_[3*leg+j]=model_.joints[id].idx_v();
                }
            }
        }
        WBCCalculation Infer(const RobotState& state,const MPCOutput& mpc_output,double kp,double kd) {
            ValidateState(state);if(!Finite(mpc_output))throw std::invalid_argument("Invalid WBC input");
            const auto q=Configuration(state),v=Velocity(state);
            pinocchio::computeAllTerms(model_,*data_,q,v);pinocchio::updateFramePlacements(model_,*data_);
            Eigen::Matrix<double,12,18> task;
            Eigen::Matrix<double,12,1> feet;
            for(int leg=0;leg<4;++leg){task.block<3,18>(3*leg,0)=Jacobian(leg);feet.segment<3>(3*leg)=data_->oMf[feet_[leg]].translation();}
            const Eigen::Matrix<double,12,1> action=kp*(Eigen::Map<const Eigen::Matrix<double,12,1>>(mpc_output.data())-feet)+kd*(Eigen::Map<const Eigen::Matrix<double,12,1>>(mpc_output.data()+12)-task*v);
            Eigen::JacobiSVD<Eigen::Matrix<double,12,18>> svd(task,Eigen::ComputeFullU|Eigen::ComputeFullV);svd.setThreshold(10.*18.*std::numeric_limits<float>::epsilon());
            const Eigen::Matrix<double,18,1> acc=svd.solve(action);
            // MuJoCo's free-base linear acceleration is world-frame; Pinocchio
            // differentiates local linear velocity. Account for d(R^T)/dt * v_world.
            Eigen::Matrix<double,18,1> dynamics_acc=acc;
            dynamics_acc.head<3>()-=v.segment<3>(3).cross(v.head<3>());
            const Eigen::Matrix<double,18,1> feedback=data_->M.selfadjointView<Eigen::Upper>()*dynamics_acc+data_->nle+task.transpose()*action;
            const Eigen::Matrix<double,18,1> force=-task.transpose()*Eigen::Map<const Eigen::Matrix<double,12,1>>(mpc_output.data()+24);
            WBCCalculation out;
            const Eigen::Quaterniond quat(state.q[3],state.q[4],state.q[5],state.q[6]);
            const Eigen::Vector3d world_acc=quat.toRotationMatrix()*acc.head<3>();
            for(int i=0;i<3;++i){out.qacc[i]=world_acc(i);out.qacc[3+i]=acc(3+i);}
            for(int i=0;i<12;++i){const int vi=v_indices_[i];const double c=mpc_output[36+i/3];
                out.torque(i)=float(c*force(vi)+(1.-c)*feedback(vi));
                out.qacc[6+i]=acc(vi);
            }
            return out;
        }
        std::pair<double,double> JointPositionLimits(int joint) const {
            const int qi=q_indices_.at(joint);
            return {model_.lowerPositionLimit(qi),model_.upperPositionLimit(qi)};
        }
    };
    std::string model_path_,backend_path_,mpx_root_,implementation_,targets_backend_;
    types::Vec3f limits_;
    double kp_=0.,kd_=0.;
    std::unique_ptr<IKWBC> ik_;
    std::unique_ptr<JaxWBC> jax_;
    std::unique_ptr<PinocchioWBC> pinocchio_;
    std::vector<std::array<double,2>> joint_limits_,model_torque_limits_;
    bool ready_=false;
public:
    WBCRunner(std::string path,std::string backend,
              const types::Vec3f& limits=types::Vec3f(40.f,40.f,65.f),
              std::string implementation="ik",std::string mpx_root="",std::string targets_backend="integration")
        :model_path_(std::move(path)),backend_path_(std::move(backend)),
         mpx_root_(std::move(mpx_root)),implementation_(std::move(implementation)),targets_backend_(std::move(targets_backend)),limits_(limits) {
        if(implementation_!="jax" && implementation_!="pinocchio" && implementation_!="ik")throw std::invalid_argument("wbc_backend must be jax, pinocchio or ik");
        if(targets_backend_!="integration" && targets_backend_!="ik")throw std::invalid_argument("wbc_targets_backend must be integration or ik");
        if(!limits_.allFinite() || (limits_.array()<=0).any())throw std::invalid_argument("Invalid WBC limits");
    }
    WBCRunner(const WBCRunner&)=delete;
    WBCRunner& operator=(const WBCRunner&)=delete;
    void Compile() {
        if(ready_)return;
        pybind11::module_::import("sys").attr("path").attr("insert")(0,backend_path_);
        if(implementation_=="ik" || targets_backend_=="ik")ik_=std::make_unique<IKWBC>(model_path_);
        if(implementation_=="ik") {
            joint_limits_=ik_->Limits(false);model_torque_limits_=ik_->Limits(true);
        } else if(implementation_=="jax") {
            jax_=std::make_unique<JaxWBC>(model_path_,mpx_root_);
            joint_limits_=jax_->Limits("joint_limits");
            model_torque_limits_=jax_->Limits("torque_limits");
        } else {
            auto metadata=pybind11::module_::import("wbc_solver").attr("model_metadata")(model_path_,mpx_root_).cast<pybind11::dict>();
            joint_limits_=metadata["joint_limits"].cast<std::vector<std::array<double,2>>>();
            model_torque_limits_=metadata["torque_limits"].cast<std::vector<std::array<double,2>>>();
            kp_=metadata["kp"].cast<double>();kd_=metadata["kd"].cast<double>();
            pinocchio_=std::make_unique<PinocchioWBC>(model_path_);
        }
        if(joint_limits_.size()!=12 || model_torque_limits_.size()!=12)
            throw std::runtime_error("Invalid WBC model dimensions");
        ready_=true;
    }
    WBCCalculation Infer(const RobotState& state,const MPCOutput& mpc_output) {
        ValidateState(state);if(!Finite(mpc_output))throw std::invalid_argument("Invalid WBC MPC output");
        for(int leg=0;leg<4;++leg)if(mpc_output[36+leg]<0 || mpc_output[36+leg]>1)throw std::invalid_argument("Invalid contact");
        if(!ready_)throw std::runtime_error("WBC is not compiled");
        auto output=implementation_=="ik" ? ik_->Infer(state,mpc_output)
                    : (jax_ ? jax_->Infer(state,mpc_output) : pinocchio_->Infer(state,mpc_output,kp_,kd_));
        if(implementation_!="ik" && targets_backend_=="ik")
            output.joint_targets=ik_->Infer(state,mpc_output).joint_targets;
        if(!output.torque.allFinite() || !Finite(output.qacc))throw std::runtime_error("Non-finite WBC output");
        for(int i=0;i<12;++i) {
            const double lower=std::max(double(-limits_(i%3)),model_torque_limits_[i][0]);
            const double upper=std::min(double(limits_(i%3)),model_torque_limits_[i][1]);
            if(lower>upper)throw std::runtime_error("Inconsistent WBC torque limits");
            output.torque(i)=float(std::clamp(double(output.torque(i)),lower,upper));
        }
        return output;
    }
    std::pair<double,double> JointPositionLimits(int joint) const {
        if(joint<0 || joint>=12 || !ready_)throw std::out_of_range("Invalid WBC joint");
        return {joint_limits_[joint][0],joint_limits_[joint][1]};
    }
};
}
