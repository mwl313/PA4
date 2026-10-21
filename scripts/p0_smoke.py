"""P0 smoke: pinned TorchRL API imports, TensorDict 2x4 op, and an 8x8 CUDA tensor op.

This proves import availability, exact installed versions and device access ONLY.
It does not prove action compatibility, a custom EnvBase contract, collector
throughput or process transport reliability. Those belong to P1-P3.
"""

import inspect
import sys

import torch

EXPECTED = {
    "torch": "2.9.1+cu128",
    "torchrl": "0.14.1",
    "tensordict": "0.14.3",
}


def check(condition: bool, message: str) -> None:
    """Fail loudly regardless of -O / PYTHONOPTIMIZE (unlike a bare assert)."""
    if not condition:
        raise SystemExit(f"P0_SMOKE_FAIL: {message}")


def main() -> int:
    import tensordict
    import torchrl
    from tensordict import TensorDict
    from torchrl.collectors import AsyncBatchedCollector
    from torchrl.envs import EnvBase
    from torchrl.modules.inference_server import InferenceServer, ProcessSlotTransport

    print("python", sys.version.split()[0])
    print("torch", torch.__version__)
    print("torch.cuda", torch.version.cuda)
    print("torchrl", getattr(torchrl, "__version__", "unknown"))
    print("tensordict", getattr(tensordict, "__version__", "unknown"))

    # Installed versions must equal the pins, not merely import.
    installed = {
        "torch": torch.__version__,
        "torchrl": getattr(torchrl, "__version__", "unknown"),
        "tensordict": getattr(tensordict, "__version__", "unknown"),
    }
    for name, expected in EXPECTED.items():
        check(
            installed[name] == expected,
            f"{name} installed={installed[name]!r} expected={expected!r}",
        )
    print("version_pins_ok", installed)

    # Required P0 API surface: must be real classes with an __init__ signature,
    # not aliases or placeholders.
    required = [
        ("TensorDict", TensorDict),
        ("EnvBase", EnvBase),
        ("AsyncBatchedCollector", AsyncBatchedCollector),
        ("InferenceServer", InferenceServer),
        ("ProcessSlotTransport", ProcessSlotTransport),
    ]
    for name, obj in required:
        check(inspect.isclass(obj), f"{name} is not a class (got {type(obj)})")
        params = inspect.signature(obj.__init__).parameters
        check(len(params) > 1, f"{name}.__init__ has no parameters")
        print("api_import_ok", name, "| module:", obj.__module__)
    check(issubclass(EnvBase, torch.nn.Module) or hasattr(EnvBase, "step"),
          "EnvBase lacks expected env interface")

    # TensorDict 2x4 operation (P0 design requirement).
    td = TensorDict({"observation": torch.zeros(2, 4)}, batch_size=[2])
    check(td["observation"].shape == (2, 4), f"shape {td['observation'].shape}")
    check(td.batch_size == torch.Size([2]), f"batch_size {td.batch_size}")
    print("tensordict_2x4_ok", tuple(td.batch_size), tuple(td["observation"].shape))
    print("P0_TENSORDICT_2X4_PASS")

    # Trivial GPU check (P0 design requirement): 8x8 CUDA tensor sum == 64.
    check(torch.cuda.is_available(), "CUDA unavailable on dev machine")
    x = torch.ones((8, 8), device="cuda")
    total = x.sum().item()
    check(total == 64.0, f"8x8 cuda sum={total}")
    check(x.device.type == "cuda", f"tensor landed on {x.device}")
    print("gpu_name", torch.cuda.get_device_name(0))
    print("cuda_8x8_sum", total)
    print("P0_CUDA_8X8_PASS")

    print("P0_TORCHRL_API_IMPORT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
