"""Actual workspace MPX WBC. Returns torque and qacc; no target integration."""
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from srbd_solver import load_config


class Solver:
    def __init__(self, model_path, input_buffer, output_buffer, mpx_root):
        config = load_config(model_path, mpx_root)
        from mpx.utils.mpc_wrapper_srbd import BatchedMPCControllerWrapper
        self.wrapper = BatchedMPCControllerWrapper(config, n_env=1)
        self.state = self.wrapper.init_state()
        self.input = np.frombuffer(input_buffer, dtype=np.float64, count=77)
        self.output = np.frombuffer(output_buffer, dtype=np.float64, count=30)
        model = mujoco.MjModel.from_xml_path(model_path)
        self.joint_limits = model.jnt_range[1:].tolist()
        self.torque_limits = model.actuator_ctrlrange.tolist()
        self.nominal = np.concatenate([config.p0, config.quat0, config.q0]).tolist()
        self._run = jax.jit(self.run)

    def run(self, inputs):
        mpc_output = inputs[37:]
        state = self.state.replace(
            grf=mpc_output[24:36][None, :], foot_ref=mpc_output[:12][None, :],
            foot_ref_dot=mpc_output[12:24][None, :], contact=mpc_output[36:][None, :])
        tau, _J, qacc = self.wrapper.whole_body_run(state, inputs[:19][None, :], inputs[19:37][None, :])
        
        return jnp.concatenate([tau[0], qacc[0]])

    def step(self):
        np.copyto(self.output, np.asarray(self._run(self.input)), casting='unsafe')


def model_metadata(model_path, mpx_root):
    """Read the shared workspace configuration without compiling the JAX WBC."""
    config = load_config(model_path, mpx_root)
    def scalar_gain(matrix):
        matrix = np.asarray(matrix)
        value = float(matrix[0, 0])
        if not np.allclose(matrix, np.eye(matrix.shape[0]) * value):
            raise ValueError("Pinocchio backend requires uniform diagonal Cartesian gains")
        return value
    model = mujoco.MjModel.from_xml_path(model_path)
    return dict(joint_limits=model.jnt_range[1:].tolist(),
                torque_limits=model.actuator_ctrlrange.tolist(),
                kp=scalar_gain(config.Kp),
                kd=scalar_gain(config.Kd))
