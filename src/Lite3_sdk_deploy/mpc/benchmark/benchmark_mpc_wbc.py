#!/usr/bin/env python3
"""Lite3 MPC/WBC benchmark: measures the running solver nodes and prints one table.

Port of the TITA benchmark_mpx.py onto the Lite3 pipeline:
MPC_CONTROL -> lite3_mpc -> MPC_OUTPUT -> lite3_wbc -> WBC_JOINTS_CMD -> rl_deploy
The nodes compute only while MPC_CONTROL is active. Unless rl_deploy is already in
MPC, this script activates them itself with its own session and zero velocity;
rl_deploy discards WBC output from other sessions, so no joint command reaches
the motors. If no simulation or robot publishes state, it starts the simulation
headless (SDK_SIM_HEADLESS=1) and stops it at the end. It does not start the
MPC/WBC nodes and writes no files.
Rates are measured on MPC_OUTPUT and WBC_JOINTS_CMD reception (1 s bins); solve
times come from the nodes' "MPC timing"/"WBC timing" /rosout windows.

    ros2 run lite3_sdk_deploy benchmark_mpc_wbc.py --duration 60
"""
import argparse
import collections
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import statistics
import sys
import threading
import time

TIMING = re.compile(r'(MPC|WBC) timing: n=(\d+), solve ms mean=([\d.]+) std=([\d.]+)')


def topic_name(namespace, name):
    return '/' + '/'.join(p for p in (namespace.strip('/'), name) if p)


def logger_matches(actual, expected):
    return actual == expected or actual.endswith('.' + expected) or actual.endswith('/' + expected)


def pooled(windows):
    """Exact mean/std over all samples from per-window (n, mean, std)."""
    n = sum(w[0] for w in windows)
    if not n:
        return None, None
    mean = sum(w[0] * w[1] for w in windows) / n
    squares = sum(w[0] * (w[2] ** 2 + w[1] ** 2) for w in windows) / n
    return mean, math.sqrt(max(0., squares - mean ** 2))


def find_sim_script():
    """Simulator source next to this package, from the source or install tree."""
    relative = Path('interface/robot/simulation/mujoco_simulation_ros2.py')
    for folder in Path(__file__).resolve().parents:
        for candidate in (folder / relative, folder / 'src/Lite3_sdk_deploy' / relative):
            if candidate.is_file():
                return candidate
    return None


def start_headless_sim(args):
    script = Path(args.sim_script) if args.sim_script else find_sim_script()
    if script is None or not script.is_file():
        return None, 'simulator script not found (pass --sim-script)'
    print(f'No simulation/robot data: starting the simulation headless (SDK_SIM_HEADLESS=1): {script}',
          flush=True)
    env = dict(os.environ, SDK_SIM_HEADLESS='1')
    process = subprocess.Popen([sys.executable, str(script)], env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               start_new_session=True)
    # Drain stderr so the simulator never blocks on a full pipe; keep the tail for errors.
    process.tail = collections.deque(maxlen=40)
    threading.Thread(target=lambda: process.tail.extend(
        line.decode(errors='replace') for line in process.stderr), daemon=True).start()
    return process, None


def stop_headless_sim(process):
    if process is None or process.poll() is not None:
        return
    print('Stopping the headless simulation.', flush=True)
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def wait_for_state(node, received, seconds, process=None):
    """Spin until all state topics have data; False on timeout or simulator exit."""
    import rclpy
    deadline = time.monotonic() + seconds
    while not all(received.values()):
        rclpy.spin_once(node, timeout_sec=.1)
        if time.monotonic() >= deadline or (process is not None and process.poll() is not None):
            return False
    return True


def preflight(args, node, received):
    """Return what is missing among robot/simulation data and the pipeline nodes."""
    import rclpy
    namespace = '/' + args.namespace.strip('/')
    required = [('lite3_mpc', namespace, 'start: ros2 launch lite3_sdk_deploy mpc_wbc.launch.py'),
                ('lite3_wbc', namespace, 'start: ros2 launch lite3_sdk_deploy mpc_wbc.launch.py')]
    print(f'Checking simulation/robot data and nodes (up to {args.preflight_timeout:g} s)...', flush=True)
    deadline = time.monotonic() + args.preflight_timeout
    while True:
        rclpy.spin_once(node, timeout_sec=.1)
        nodes = set(node.get_node_names_and_namespaces())
        missing = [f'{topic}: no data (start the simulation or the robot)'
                   for topic, ok in received.items() if not ok]
        missing += [f'node {ns.rstrip("/")}/{name} ({hint})' for name, ns, hint in required
                    if (name, ns) not in nodes]
        if not missing and time.monotonic() >= deadline - args.preflight_timeout + .5:
            return missing  # At least 0.5 s spent, so an existing MPC_CONTROL owner is seen.
        if time.monotonic() >= deadline:
            return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--duration', type=float, default=60,
                        help='Seconds measured after the first MPC/WBC output (default 60)')
    parser.add_argument('--wait-timeout', type=float, default=180,
                        help='Maximum seconds waiting for MPC/WBC output, e.g. while compiling (default 180)')
    parser.add_argument('--preflight-timeout', type=float, default=5,
                        help='Seconds to find simulation/robot data and nodes (default 5)')
    parser.add_argument('--namespace', default='', help='pipeline_namespace of mpc_wbc.launch.py')
    parser.add_argument('--no-sim', action='store_true',
                        help='Never start the headless simulation; require existing state data')
    parser.add_argument('--sim-script', help='Simulator script (default: found in the source tree)')
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in
               (args.duration, args.wait_timeout, args.preflight_timeout)):
        parser.error('duration and timeouts must be positive')

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.executors import ExternalShutdownException
    from rcl_interfaces.msg import Log
    from std_msgs.msg import Float64MultiArray
    from drdds.msg import ImuData, JointsData

    rclpy.init(args=[])
    node = rclpy.create_node('lite3_mpc_wbc_benchmark', enable_rosout=False)
    qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
    received = {}
    base = [None]  # Latest (wall time, base stamp), for the simulation real-time factor.
    def state(name, msg):
        received[name] = True
        if name == '/BASE_DATA' and len(msg.data) == 7:
            base[0] = (time.monotonic(), msg.data[0])
    for name, kind in (('/BASE_DATA', Float64MultiArray), ('/IMU_DATA', ImuData),
                       ('/JOINTS_DATA', JointsData)):
        received[name] = False
        node.create_subscription(kind, name, lambda m, n=name: state(n, m), qos)

    seen = {'MPC': False, 'WBC': False}
    # Own session ID, distinct from rl_deploy's (microseconds since epoch, like MPCPipelineClient).
    epoch = float(time.time_ns() // 1000)
    driving = [False]
    foreign = []
    def control(msg):
        if len(msg.data) == 5 and msg.data[0] != epoch and msg.data[1] == 1:
            foreign.append(msg.data[0])
    control_topic = topic_name(args.namespace, 'MPC_CONTROL')
    control_pub = node.create_publisher(Float64MultiArray, control_topic, 1)
    node.create_subscription(Float64MultiArray, control_topic, control, qos)
    def heartbeat(active=True):
        msg = Float64MultiArray()
        msg.data = [epoch, 1. if active else 0., 0., 0., 0.]
        control_pub.publish(msg)
    # The nodes drop the session after 150 ms without a heartbeat.
    node.create_timer(.05, lambda: driving[0] and heartbeat())
    counts = {'MPC': 0, 'WBC': 0}
    windows = {'MPC': [], 'WBC': []}
    measuring = [False]
    def output(component):
        def callback(_msg):
            seen[component] = True
            if measuring[0]:
                counts[component] += 1
        return callback
    def log(msg):
        match = TIMING.search(msg.msg)
        if (measuring[0] and match and (logger_matches(msg.name, 'lite3_mpc')
                                        or logger_matches(msg.name, 'lite3_wbc'))):
            windows[match.group(1)].append((int(match.group(2)), float(match.group(3)),
                                            float(match.group(4))))
    for component, topic in (('MPC', 'MPC_OUTPUT'), ('WBC', 'WBC_JOINTS_CMD')):
        node.create_subscription(Float64MultiArray, topic_name(args.namespace, topic),
                                 output(component), qos)
    node.create_subscription(Log, '/rosout', log, qos)

    status = 0
    rates = {'MPC': [], 'WBC': []}
    sim = None
    source = 'existing simulation/robot'
    base_start = base_end = None
    try:
        if not wait_for_state(node, received, 2.) and not args.no_sim:
            sim, error = start_headless_sim(args)
            if error:
                print(f'Benchmark cannot start: {error}.', flush=True)
                return 2
            if not wait_for_state(node, received, 30., sim):
                time.sleep(.2)  # Let the drain thread collect the last lines.
                reason = (''.join(sim.tail).strip() or f'exit code {sim.returncode}') \
                    if sim.poll() is not None else 'no state data within 30 s'
                print(f'Headless simulation failed: {reason}', flush=True)
                return 2
            source = 'headless simulation started by this script'
            print('Headless simulation running.', flush=True)
        elif all(received.values()):
            print('Using the simulation/robot already publishing state (not starting one).', flush=True)
        missing = preflight(args, node, received)
        if missing:
            print('Benchmark cannot start, missing:', flush=True)
            for item in missing:
                print(f'  - {item}', flush=True)
            return 2
        if foreign:
            print('rl_deploy is already in MPC: measuring its session without sending anything.',
                  flush=True)
        else:
            print('Activating the MPC/WBC nodes with a benchmark session (zero velocity).', flush=True)
            driving[0] = True
        deadline = time.monotonic() + args.wait_timeout
        while not (seen['MPC'] and seen['WBC']):
            rclpy.spin_once(node, timeout_sec=.1)
            if time.monotonic() >= deadline:
                print(f'No MPC/WBC output within {args.wait_timeout:g} s: '
                      'are the nodes still compiling (MPC_READY/WBC_READY)?', flush=True)
                return 2
        foreign.clear()
        print(f'MPC/WBC output received: measuring for {args.duration:g} s...', flush=True)
        measuring[0] = True
        start = bin_start = time.monotonic()
        base_start = base[0]
        while time.monotonic() - start < args.duration:
            rclpy.spin_once(node, timeout_sec=.01)
            if driving[0] and foreign:
                print('rl_deploy entered MPC during the benchmark: stopping to leave it the pipeline.',
                      flush=True)
                driving[0] = False
                status = 3
                break
            now = time.monotonic()
            if now - bin_start >= 1.:
                for component in rates:
                    rates[component].append(counts[component] / (now - bin_start))
                    counts[component] = 0
                bin_start = now
        base_end = base[0]
    except (KeyboardInterrupt, ExternalShutdownException):
        print('\nInterrupted: results so far.', flush=True)
        status = 130
    finally:
        if driving[0]:
            heartbeat(active=False)  # End the benchmark session explicitly.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        stop_headless_sim(sim)

    def fmt(value, digits):
        return 'n/a' if value is None else f'{value:.{digits}f}'
    print(f"\n{'Module':<8}{'Rate mean [Hz]':>16}{'Rate std [Hz]':>15}"
          f"{'Solve mean [ms]':>17}{'Solve std [ms]':>16}")
    for component in ('MPC', 'WBC'):
        values = rates[component]
        hz = statistics.mean(values) if values else None
        hz_std = statistics.pstdev(values) if len(values) > 1 else None
        solve, solve_std = pooled(windows[component])
        print(f'{component:<8}{fmt(hz, 2):>16}{fmt(hz_std, 2):>15}'
              f'{fmt(solve, 3):>17}{fmt(solve_std, 3):>16}')
    print(f'State source: {source}.')
    if base_start and base_end and base_end[0] > base_start[0]:
        rtf = (base_end[1] - base_start[1]) / (base_end[0] - base_start[0])
        print(f'Simulation real-time factor: {rtf:.2f} (MPC runs at its rate on the simulation '
              f'clock, so its wall-clock rate is about mpc_frequency x {rtf:.2f}).')
    if not all(windows.values()):
        print('Solve time n/a: no 5 s timing window from that node; measure longer or check /rosout.')
    return status


if __name__ == '__main__':
    sys.exit(main())
