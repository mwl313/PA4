"""D0: capture the full environment + source freeze for P4.5.

Writes docs/phases/P4.5/ENVIRONMENT.json and an evidence copy. No training code
is touched; this only reads.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

REPO = "/home/pa4/projects/PokeAgent4.0"
OUT = os.path.join(REPO, "docs/phases/P4.5")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, shell=True, cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import torch
    import tensordict
    import torchrl

    env = {
        "phase": "P4.5",
        "design_id": "PA4-P45-DIAG-DESIGN-v1.0",
        "captured_utc": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),

        "source": {
            "repo": REPO,
            "head": sh("git rev-parse HEAD"),
            "branch": sh("git branch --show-current"),
            "worktree_clean": sh("git status --porcelain | wc -l") == "0",
            "pa3_reference": sh("git -C /home/pa4/projects/_pa3_ref rev-parse HEAD"),
            "pa3_dirty_files": int(sh("git -C /home/pa4/projects/_pa3_ref status --porcelain | wc -l")),
            "uv_lock_sha256": sha256_file(os.path.join(REPO, "uv.lock")),
            "uv_lock_check": sh("PATH=/home/pa4/.local/bin:$PATH uv lock --check"),
        },

        "python": {
            "version": sys.version.split()[0],
            "torch": torch.__version__,
            "torchrl": torchrl.__version__,
            "tensordict": tensordict.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },

        "host": {
            "platform": platform.platform(),
            "kernel": sh("uname -r"),
            "cpu_model": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(),
            "logical_cores": int(sh("nproc")),
            "mem_total_kb": int(sh("grep MemTotal /proc/meminfo | awk '{print $2}'")),
            "wsl": "microsoft" in sh("uname -r").lower(),
        },

        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
            if torch.cuda.is_available() else None,
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count
            if torch.cuda.is_available() else None,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))
            if torch.cuda.is_available() else None,
            "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,memory.total,"
                             "clocks.sm,power.draw,temperature.gpu,pstate "
                             "--format=csv,noheader 2>/dev/null || "
                             "cat /usr/lib/wsl/lib/nvidia-smi >/dev/null 2>&1"),
        },

        "thread_env": {
            k: os.environ.get(k) for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
             "CUDA_LAUNCH_BLOCKING", "TORCH_NUM_THREADS", "PA4_P3_STATS_PATH",
             "PA4_POLICY_SNAPSHOT")
        },

        "torch_flags": {
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        },
    }

    # Hash the protected P4 evidence we must not modify.
    p4 = os.path.join(REPO, "docs/phases/P4")
    evidence = {}
    for rel in ("FULL_STACK_BASELINE.csv", "RECEIPT.json", "FINAL_REPORT.md",
                "evidence/full_run.json", "CHECKPOINT_RECOVERY.md"):
        path = os.path.join(p4, rel)
        if os.path.exists(path):
            evidence[rel] = {"sha256": sha256_file(path),
                             "bytes": os.path.getsize(path)}
    env["protected_p4_evidence"] = evidence

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ENVIRONMENT.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)
    os.makedirs(os.path.join(OUT, "evidence"), exist_ok=True)
    with open(os.path.join(OUT, "evidence", "environment_freeze.json"), "w",
              encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)

    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
