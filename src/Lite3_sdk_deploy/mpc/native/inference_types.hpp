#pragma once
#include "common_types.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lite3_mpc {
// MuJoCo layout: xyz, quaternion wxyz, joints; world linear/body angular velocity.
struct RobotState {
    std::array<double, 19> q{};
    std::array<double, 18> dq{};
};
using MPCPlan = std::array<double, 40>; // feet xyz, feet velocity, GRF, contacts
struct WBCOutput {
    types::VecXf torque = types::VecXf::Zero(12);
    types::VecXf q_des = types::VecXf::Zero(12);
    types::VecXf dq_des = types::VecXf::Zero(12);
};
template<class Container> bool Finite(const Container& values) {
    return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}
inline void ValidateState(const RobotState& state) {
    if (!Finite(state.q) || !Finite(state.dq)) throw std::invalid_argument("Non-finite robot state");
    double norm = 0;
    for (int i=3; i<7; ++i) norm += state.q[i]*state.q[i];
    if (std::abs(norm-1.) > .01) throw std::invalid_argument("Invalid base quaternion");
}
}
