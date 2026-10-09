"""Numerical JAX backend only: no ROS, per-tick MuJoCo, IK, or Python lists.

C++ owns the input/output storage. Memoryviews are bound once and viewed by
NumPy without copying; JAX still transfers inputs and results as required.
"""
import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

MPX_ROOT = Path(__file__).resolve().parent / 'upstream'
if not (MPX_ROOT / 'mpx/utils/mpc_wrapper_srbd.py').is_file():
    raise RuntimeError('MPX checkout missing from mpc/upstream; install it and rebuild Lite3.')
sys.path.insert(0, str(MPX_ROOT))
mpx_package = ModuleType('mpx')
mpx_package.__path__ = [str(MPX_ROOT / 'mpx')]
mpx_package.__package__ = 'mpx'
sys.modules['mpx'] = mpx_package

import gc
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mpx.utils.mpc_wrapper_srbd import BatchedMPCControllerWrapper


def load_config(model_path):
    """Adapt the upstream Lite3 config to the SDK MuJoCo model."""
    source = MPX_ROOT / 'mpx/config/config_lite3.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and
        (node.module or '').startswith('mujoco_playground')
    ) and not (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'model_path'
            for target in node.targets
        )
    )]
    values = {'__name__': 'lite3_sdk_config', 'model_path': str(model_path)}
    exec(compile(tree, str(source), 'exec'), values)
    cfg = SimpleNamespace(**{k: v for k, v in values.items() if not k.startswith('__')})
    cfg.contact_frame = [f'{leg}_FOOT_collision' for leg in ('FL', 'FR', 'HL', 'HR')]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = np.concatenate([cfg.p0, cfg.quat0, cfg.q0])
    mujoco.mj_forward(model, data)
    feet = np.array([data.geom_xpos[model.geom(name).id] for name in cfg.contact_frame])
    feet[:, :2] -= data.qpos[:2]
    feet[:, 2] = 0
    cfg.p_legs0 = jnp.asarray(feet.reshape(-1))
    cfg.mass = float(model.body_subtreemass[1])

    com = data.subtree_com[1]
    inertia = np.zeros((3, 3))
    for i in range(1, model.nbody):
        rotation = data.ximat[i].reshape(3, 3)
        offset = data.xipos[i] - com
        inertia += rotation @ np.diag(model.body_inertia[i]) @ rotation.T
        inertia += model.body_mass[i] * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    cfg.inertia = jnp.asarray(inertia)
    cfg.max_lin_acc = 0.8
    cfg.max_yaw_acc = 1.5
    return cfg


class Solver:
    def __init__(self, model_path, input_buffer, output_buffer):
        self.config = load_config(model_path)
        self.wrapper = BatchedMPCControllerWrapper(self.config, n_env=1)
        self.nominal = np.concatenate([self.config.p0, self.config.quat0, self.config.q0])
        self.mpc_frequency = float(self.config.mpc_frequency)
        self.whole_body_frequency = float(self.config.whole_body_frequency)
        self.input = np.frombuffer(input_buffer, dtype=np.float64, count=32)
        self.output = np.frombuffer(output_buffer, dtype=np.float64, count=40)
        self.state = self.wrapper.init_state()
        self._run = jax.jit(self.run)

    def run(self, state, inputs, reset):
        x = inputs[:13][None, :]
        feet = inputs[13:25][None, :]
        contacts = inputs[25:29][None, :]
        command = jnp.array([inputs[29], inputs[30], 0., 0., 0., inputs[31],
                             self.config.robot_height])[None, :]
        def initial(_):
            return self.wrapper.init_state(x).replace(liftoff=feet, foot_ref=feet)
        state = jax.lax.cond(reset, initial, lambda _: state, operand=None)
        state = self.wrapper.run(state, x, command, feet, contacts)
        # Upstream init uses float contacts, while the gait timer returns ints.
        # Keep a stable pytree dtype across reset and continuing JIT calls.
        state = state.replace(contact=state.contact.astype(jnp.float32))
        values = jnp.concatenate([state.foot_ref[0], state.foot_ref_dot[0],
                                  state.grf[0], state.contact[0]])
        return state, values

    def step(self, reset):
        self.state, values = self._run(self.state, self.input, reset)
        # One synchronized device-to-host result; no tolist()/per-scalar conversion.
        np.copyto(self.output, np.asarray(values), casting='unsafe')

    def finish_warmup(self):
        gc.collect()
        gc.freeze()
