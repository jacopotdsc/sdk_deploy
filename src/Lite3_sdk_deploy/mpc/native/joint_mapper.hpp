#pragma once
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <array>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace lite3_mpc {
class JointMapper {
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model_{nullptr, mj_deleteModel};
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data_{nullptr, mj_deleteData};
    std::unique_ptr<mjData, decltype(&mj_deleteData)> ik_{nullptr, mj_deleteData};
    std::array<int,4> feet_{};
    std::array<double,54> jac_{}, rotation_{};
    using Jacobian = Eigen::Matrix<double,3,18,Eigen::RowMajor>;
public:
    explicit JointMapper(const std::string& path) {
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
    // 13 SRBD state + 12 foot positions + 4 measured contacts + 3 velocity requests.
    void Inputs(std::array<double,32>& out) const {
        std::copy_n(data_->qpos,7,out.begin()); std::copy_n(data_->qvel,6,out.begin()+7);
        std::fill(out.begin()+25,out.begin()+29,0.);
        for(int leg=0;leg<4;++leg) std::copy_n(data_->geom_xpos+3*feet_[leg],3,out.begin()+13+3*leg);
        for(int c=0;c<data_->ncon;++c) for(int leg=0;leg<4;++leg)
            if(data_->contact[c].dist<=0 && (data_->contact[c].geom1==feet_[leg] || data_->contact[c].geom2==feet_[leg]))
                out[25+leg]=1.;
    }
    // Solver output: foot positions/velocities, GRF, planned contacts (40 doubles).
    std::array<double,36> Map(const std::array<double,40>& plan) {
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
            const Eigen::Map<const Eigen::Vector3d> force(plan.data()+24+3*leg);
            Eigen::Map<Eigen::Vector3d>(out.data()+3*leg) -= local.transpose()*(force*plan[36+leg]);
            const Eigen::Map<const Eigen::Vector3d> ref_velocity(plan.data()+12+3*leg);
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(local,Eigen::ComputeFullU|Eigen::ComputeFullV);
            svd.setThreshold(1e-3);
            Eigen::Map<Eigen::Vector3d>(out.data()+24+3*leg)=svd.solve(ref_velocity-jac.block<3,6>(0,0)*base_velocity);
            for(int iteration=0;iteration<6;++iteration) {
                mj_forward(model_.get(),ik_.get());
                mj_jac(model_.get(),ik_.get(),jac_.data(),rotation_.data(),ik_->geom_xpos+3*geom,body);
                Eigen::Vector3d error=Eigen::Map<const Eigen::Vector3d>(plan.data()+3*leg)-Eigen::Map<const Eigen::Vector3d>(ik_->geom_xpos+3*geom);
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
};
}
