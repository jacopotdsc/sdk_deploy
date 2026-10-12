"""Export a trusted Brax PPO checkpoint; training settings must be supplied explicitly."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from brax.training.networks import MLP
from onnx import TensorProto, helper, numpy_helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--activation", required=True, choices=["swish", "elu", "relu", "tanh"])
    parser.add_argument("--distribution", required=True, choices=["tanh_normal", "normal"])
    parser.add_argument("--action-scale", required=True, type=float)
    parser.add_argument("--default-pose", required=True, type=float, nargs=12)
    args = parser.parse_args()
    # Only load a checkpoint supplied by the operator: pickle is executable input.
    normalizer, actor, _critic = pickle.loads(args.checkpoint.read_bytes())
    mean = np.asarray(normalizer.mean["state"], dtype=np.float32)
    std = np.asarray(normalizer.std["state"], dtype=np.float32)
    if mean.shape != (48,) or std.shape != (48,) or not np.all(std > 0):
        raise ValueError("Expected finite 48-dimensional actor normalization")
    layers = actor["params"]
    if set(layers) != {f"hidden_{i}" for i in range(len(layers))}:
        raise ValueError("Unsupported actor structure")
    nodes, initializers = [], []

    def constant(name, value):
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite constant: {name}")
        initializers.append(numpy_helper.from_array(array, name))

    constant("mean", mean)
    constant("std", std)
    nodes.extend([helper.make_node("Sub", ["obs", "mean"], ["centered"]),
                  helper.make_node("Div", ["centered", "std"], ["normalized"])])
    previous, sizes = "normalized", []
    for i in range(len(layers)):
        layer = layers[f"hidden_{i}"]
        weight, bias = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
        sizes.append(bias.size)
        constant(f"weight_{i}", weight)
        constant(f"bias_{i}", bias)
        dense = f"dense_{i}"
        nodes.append(helper.make_node("Gemm", [previous, f"weight_{i}", f"bias_{i}"], [dense]))
        previous = dense
        if i < len(layers) - 1:
            previous = f"activation_{i}"
            if args.activation == "swish":
                nodes.append(helper.make_node("Sigmoid", [dense], [f"sigmoid_{i}"]))
                nodes.append(helper.make_node("Mul", [dense, f"sigmoid_{i}"], [previous]))
            else:
                nodes.append(helper.make_node({"elu": "Elu", "relu": "Relu", "tanh": "Tanh"}[args.activation], [dense], [previous]))
    if sizes[-1] != 24:
        raise ValueError("Expected 12 actor means and 12 distribution scale parameters")
    for name, value in [("starts", [0]), ("ends", [12]), ("axes", [1])]:
        initializers.append(numpy_helper.from_array(np.array(value, dtype=np.int64), name))
    nodes.append(helper.make_node("Slice", [previous, "starts", "ends", "axes"], ["action_mean"]))
    nodes.append(helper.make_node("Tanh" if args.distribution == "tanh_normal" else "Identity", ["action_mean"], ["actions"]))
    graph = helper.make_graph(nodes, "Lite3Residual", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 48])],
                              [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 12])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    helper.set_model_props(model, {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                  "activation": args.activation, "distribution": args.distribution,
                                  "action_scale": str(args.action_scale), "default_pose": json.dumps(args.default_pose),
                                  "linvel_frame": "body", "observation_order": "linvel,gyro,gravity,joint_delta,joint_vel,last_act,command"})
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    activation = {"swish": jax.nn.swish, "elu": jax.nn.elu, "relu": jax.nn.relu, "tanh": jnp.tanh}[args.activation]
    reference = MLP(layer_sizes=sizes, activation=activation)
    rng, maximum = np.random.default_rng(0), 0.0
    for _ in range(256):
        obs = (mean + std * rng.normal(size=(1, 48))).astype(np.float32)
        expected = reference.apply(actor, (jnp.asarray(obs) - mean) / std)[:, :12]
        if args.distribution == "tanh_normal":
            expected = jnp.tanh(expected)
        actual = session.run(["actions"], {"obs": obs})[0]
        np.testing.assert_allclose(actual, np.asarray(expected), atol=2e-5, rtol=2e-5)
        maximum = max(maximum, float(np.max(np.abs(actual - np.asarray(expected)))))
    # Validate before replacing the deployment model.
    onnx.save(model, args.output)
    print(f"Exported {args.output}; 256 Flax/ONNX comparisons, maximum absolute error {maximum:.9g}")


if __name__ == "__main__":
    main()
