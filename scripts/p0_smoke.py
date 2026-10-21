"""P0 smoke: pinned TorchRL API imports, TensorDict 2x4 op, and an 8x8 CUDA tensor op.

This proves import availability and device access ONLY. It does not prove action
compatibility, a custom EnvBase contract, collector throughput or process transport
reliability. Those belong to P1-P3.
"""

import sys

import torch


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

    # Required P0 API surface must be importable identifiers, not merely present modules.
    for name, obj in [
        ("TensorDict", TensorDict),
        ("EnvBase", EnvBase),
        ("AsyncBatchedCollector", AsyncBatchedCollector),
        ("InferenceServer", InferenceServer),
        ("ProcessSlotTransport", ProcessSlotTransport),
    ]:
        assert obj is not None, f"{name} is None"
        print("api_import_ok", name)

    # TensorDict 2x4 operation (P0 design requirement).
    td = TensorDict({"observation": torch.zeros(2, 4)}, batch_size=[2])
    assert td["observation"].shape == (2, 4), td["observation"].shape
    assert td.batch_size == torch.Size([2]), td.batch_size
    print("tensordict_2x4_ok", tuple(td.batch_size), tuple(td["observation"].shape))
    print("P0_TENSORDICT_2X4_PASS")

    # Trivial GPU check (P0 design requirement): 8x8 CUDA tensor sum == 64.
    assert torch.cuda.is_available(), "CUDA unavailable on dev machine"
    x = torch.ones((8, 8), device="cuda")
    total = x.sum().item()
    assert total == 64.0, total
    print("gpu_name", torch.cuda.get_device_name(0))
    print("cuda_8x8_sum", total)
    print("P0_CUDA_8X8_PASS")

    print("P0_TORCHRL_API_IMPORT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
