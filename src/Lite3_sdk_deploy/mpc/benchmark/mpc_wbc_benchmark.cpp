// Offline MPC/WBC timing benchmark for Lite3; no ROS topics, robot or simulator.
// Port of the TITA mpx_benchmark.py onto the production MPCRunner/WBCRunner, so
// every MPC kinematics and WBC backend is measured with the code the nodes run.
// MPC is measured first, then WBC with the last valid MPC output held fixed.
// Both loops are unpaced; requested rates define budgets, not execution cadence.
//
//   ros2 run lite3_sdk_deploy mpc_wbc_benchmark
//   ros2 run lite3_sdk_deploy mpc_wbc_benchmark --mpc-hz 50 --wbc-hz 100 --wbc-backend jax
#include "mpc_runner.hpp"
#include "wbc_runner.hpp"
#include <ament_index_cpp/get_package_prefix.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;
namespace py = pybind11;
using Clock = std::chrono::steady_clock;

namespace {
struct Distribution { double mean, median, p95, p99, max; };

double Percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double index = fraction * (values.size() - 1);
    const size_t low = size_t(index), high = std::min(low + 1, values.size() - 1);
    return values[low] + (values[high] - values[low]) * (index - low);
}

Distribution Summarize(const std::vector<double>& values) {
    return {std::accumulate(values.begin(), values.end(), 0.) / values.size(),
            Percentile(values, .5), Percentile(values, .95), Percentile(values, .99),
            *std::max_element(values.begin(), values.end())};
}

double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Same search as launch/mpc_wbc.launch.py: the MPX source checkout is not installed.
std::string FindMpxRoot() {
    std::vector<fs::path> starts{fs::current_path()};
    std::error_code error;
    const auto executable = fs::read_symlink("/proc/self/exe", error);
    if (!error) starts.push_back(executable.parent_path());
    for (auto folder : starts) {
        for (;;) {
            for (const auto& relative : {fs::path("mpc/upstream"), fs::path("src/Lite3_sdk_deploy/mpc/upstream")}) {
                const auto candidate = folder / relative;
                if (fs::is_regular_file(candidate / "mpx/utils/mpc_wrapper_srbd.py")) return candidate.string();
            }
            if (folder == folder.parent_path()) break;
            folder = folder.parent_path();
        }
    }
    return {};
}

void Usage() {
    std::cout << "usage: mpc_wbc_benchmark [--steps N] [--mpc-hz F] [--wbc-hz F]\n"
                 "         [--mpc-kinematics mujoco|pinocchio] [--wbc-backend ik|jax|pinocchio]\n"
                 "         [--wbc-targets integration|ik] [--model PATH] [--mpx-root PATH]\n"
                 "Defaults: 2000 steps, 50/100 Hz, mujoco, ik, integration (as mpc_wbc.launch.py).\n";
}

void PrintComponent(const char* name, const char* unit, const std::vector<double>& samples, double hz) {
    const auto d = Summarize(samples);
    const double period = 1000. / hz;
    const auto over = std::count_if(samples.begin(), samples.end(), [&](double v) { return v > period; });
    std::cout << '\n' << name << ": " << samples.size() << ' ' << unit << "; budget rate " << std::defaultfloat << hz << std::fixed
              << " Hz (period " << std::setprecision(2) << period << std::setprecision(3) << " ms)\n"
              << "  compute ms: mean " << d.mean << " | median " << d.median << " | p95 " << d.p95
              << " | p99 " << d.p99 << " | max " << d.max << '\n'
              << "  p95 uses " << std::setprecision(1) << 100. * d.p95 / period << "% of period; "
              << "over-period samples " << over << '/' << samples.size() << "; capacity mean/p95 "
              << 1000. / d.mean << '/' << 1000. / d.p95 << " Hz\n" << std::setprecision(3);
}
}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> options{
        {"--steps", "2000"}, {"--mpc-hz", "50"}, {"--wbc-hz", "100"},
        {"--mpc-kinematics", "mujoco"}, {"--wbc-backend", "ik"}, {"--wbc-targets", "integration"},
        {"--model", ""}, {"--mpx-root", ""}};
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "-h" || key == "--help") { Usage(); return 0; }
        if (!options.count(key) || i + 1 >= argc) { Usage(); return 2; }
        options[key] = argv[++i];
    }
    int steps = 0; double mpc_hz = 0., wbc_hz = 0.;
    try {
        steps = std::stoi(options["--steps"]);
        mpc_hz = std::stod(options["--mpc-hz"]); wbc_hz = std::stod(options["--wbc-hz"]);
    } catch (const std::exception&) { Usage(); return 2; }
    if (steps < 2) { std::cerr << "--steps must be at least 2\n"; return 2; }
    if (!(std::isfinite(mpc_hz) && mpc_hz > 0 && std::isfinite(wbc_hz) && wbc_hz > 0)) {
        std::cerr << "--mpc-hz and --wbc-hz must be positive\n"; return 2;
    }
    const auto folder = ament_index_cpp::get_package_prefix("lite3_sdk_deploy") + "/lib/lite3_sdk_deploy";
    const auto model = options["--model"].empty()
        ? folder + "/Lite3_description/lite3_mjcf/mjcf/Lite3.xml" : options["--model"];
    const auto backend = folder + "/mpc";
    const auto mpx_root = options["--mpx-root"].empty() ? FindMpxRoot() : options["--mpx-root"];
    if (mpx_root.empty()) { std::cerr << "MPX source checkout not found: pass --mpx-root\n"; return 2; }

    std::cout << "NODE NAME: none (offline C++ benchmark)\n"
                 "  Publishes to topics:\n    - None\n"
                 "  Subscribes to topics:\n    - None\n"
              << "Model: " << model << "\nMPX root: " << mpx_root << '\n'
              << "MPC kinematics: " << options["--mpc-kinematics"] << "; WBC backend: "
              << options["--wbc-backend"] << ", targets: " << options["--wbc-targets"] << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    py::scoped_interpreter interpreter{};
    try {
        lite3_mpc::MPCRunner mpc(model, backend, mpc_hz, mpx_root, options["--mpc-kinematics"]);
        lite3_mpc::WBCRunner wbc(model, backend, types::Vec3f(40.f, 40.f, 65.f),
                                 options["--wbc-backend"], mpx_root, options["--wbc-targets"]);
        std::cout << "Compiling MPC and WBC independently; cold-start compilation is excluded..." << std::endl;
        auto start = Clock::now(); mpc.Compile(); const double mpc_compile = Milliseconds(start) / 1000.;
        start = Clock::now(); wbc.Compile(); const double wbc_compile = Milliseconds(start) / 1000.;
        const auto jax = py::module_::import("jax");
        std::cout << "JAX backend: " << py::str(jax.attr("default_backend")()).cast<std::string>()
                  << " (" << py::str(jax.attr("devices")()).cast<std::string>() << ")" << std::endl;

        // Static nominal posture from the same config the runners warm up with.
        const auto config = py::module_::import("srbd_solver").attr("load_config")(model, mpx_root);
        const auto numpy = py::module_::import("numpy");
        const auto nominal = numpy.attr("concatenate")(py::make_tuple(
            config.attr("p0"), config.attr("quat0"), config.attr("q0"))).attr("tolist")().cast<std::vector<double>>();
        if (nominal.size() != 19) throw std::runtime_error("Invalid nominal posture");
        lite3_mpc::RobotState state;
        std::copy(nominal.begin(), nominal.end(), state.q.begin());
        const std::array<double, 3> command{};

        std::vector<double> mpc_ms, wbc_ms;
        mpc_ms.reserve(steps); wbc_ms.reserve(steps);
        lite3_mpc::MPCOutput plan{};
        std::cout << "Timing MPC separately: " << steps << " plans (kinematics + JAX solve)..." << std::endl;
        for (int i = 0; i < steps; ++i) {
            start = Clock::now();
            plan = mpc.Infer(state, command, i == 0);  // Fresh receding-horizon state, as on activation.
            mpc_ms.push_back(Milliseconds(start));
        }
        std::cout << "Timing WBC separately: " << steps << " solves with fixed MPC output..." << std::endl;
        const double dt = 1. / wbc_hz;
        for (int i = 0; i < steps; ++i) {
            start = Clock::now();
            const auto calculation = wbc.Infer(state, plan);
            // Same qacc integration as WBCNode::Tick when targets come from integration.
            lite3_mpc::WBCOutput output; output.torque = calculation.torque;
            if (calculation.joint_targets) {
                output.q_des = calculation.joint_targets->q_des;
                output.dq_des = calculation.joint_targets->dq_des;
            } else for (int j = 0; j < 12; ++j) {
                const auto limits = wbc.JointPositionLimits(j);
                const double qacc = calculation.qacc[6 + j];
                output.q_des(j) = float(std::clamp(state.q[7 + j] + state.dq[6 + j] * dt + .5 * qacc * dt * dt,
                                                   limits.first, limits.second));
                output.dq_des(j) = float(std::clamp(state.dq[6 + j] + qacc * dt, -10., 10.));
            }
            wbc_ms.push_back(Milliseconds(start));
            if (!output.torque.allFinite() || !output.q_des.allFinite() || !output.dq_des.allFinite())
                throw std::runtime_error("non-finite WBC result at tick " + std::to_string(i));
        }

        std::cout << "Compilation MPC / WBC: " << std::setprecision(2) << mpc_compile << " / "
                  << wbc_compile << " s (excluded from timing)" << std::setprecision(3) << '\n';
        PrintComponent("MPC", "plans", mpc_ms, mpc_hz);
        PrintComponent("WBC", "solves", wbc_ms, wbc_hz);
        const double cpu = (Summarize(mpc_ms).mean * mpc_hz + Summarize(wbc_ms).mean * wbc_hz) / 10.;
        std::cout << std::setprecision(1)
                  << "\nEstimated aggregate CPU demand at requested rates: " << cpu
                  << "% (one core = 100%, compute only; extrapolated from separate runs)\n"
                     "Invalid MPC/WBC results: 0\n"
                     "MPC timing includes MuJoCo/Pinocchio input preparation (inside MPCRunner::Infer).\n"
                     "Unpaced static-posture benchmark: reports solver compute time, not ROS/DDS, "
                     "scheduler jitter, simulation tracking, or end-to-end latency.\n";
    } catch (const std::exception& error) {
        std::cerr << "[benchmark] " << error.what() << '\n';
        return 1;
    }
    return 0;
}
