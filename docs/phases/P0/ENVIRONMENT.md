# P0-A — Development Host Environment

Document ID: `PA4-P0-ENV-v1.0`
Scope: P0-A (host/WSL2/GPU preflight). Observed on the actual development workstation.

---

## 1. Host summary

| Item | Measured |
|---|---|
| Host OS | Windows 11 (Windows `10.0.26300.9711` reported by `wsl --version`) |
| Virtualization | WSL2, `wsl --version` → WSL `2.7.12.0`, kernel `6.18.33.2-2`, WSLg `1.0.73.2` |
| Distro | Ubuntu `24.04.3 LTS (Noble Numbat)`, `VERSION_ID="24.04"` |
| Linux kernel (guest) | `6.18.35.2-microsoft-standard-WSL2+` x86_64 |
| Hostname | `DESKTOP-CV81OHM` |
| NVIDIA driver (Windows) | `610.88`, CUDA UMD `13.3` |
| GPU | `NVIDIA GeForce RTX 4060 Ti`, `8188 MiB` (8 GB), bus `00000000:01:00.0`, WDDM |

GPU is exposed to WSL2 through the standard Windows WSL passthrough
(`nvidia-smi` works inside the distro). **No Linux NVIDIA kernel driver was
installed inside WSL**, as required.

### `nvidia-smi` inside WSL2

```
NVIDIA GeForce RTX 4060 Ti, 610.88, 8188 MiB
```

---

## 2. Toolchains (inside WSL2)

| Tool | Version | Notes |
|---|---|---|
| Python | `3.12.3` | system interpreter `/usr/bin/python3.12` |
| Rust / Cargo | not present at P0-A time | P0 does not require a Rust build; PA3 pins Rust `1.90.0` (`rust-toolchain.toml`) for P1/P2 |
| Git | `2.43.0` | installed during P0-A (was absent in base image) |
| gcc / make | `13.3.0` / `4.3` | build-essential, needed for native wheels |
| curl | `8.5.0` | installed during P0-A |
| uv | `0.13.4` | dependency/environment manager (`~/.local/bin`) |

---

## 3. Workspace

| Item | Measured |
|---|---|
| PA4 repo | `/root/projects/PokeAgent4.0` (WSL ext4 root filesystem, **not** `/mnt/c`) |
| Filesystem for `/` | `/dev/sdc`, 1007G total, ~956G available |
| PA3 reference | `/root/projects/_pa3_ref/pa3` (read-only intent, separate directory) |
| Windows working copy | `C:\Users\<user>\Desktop\PA4` (source design docs only) |

Running Linux builds and Python tests from the WSL ext4 filesystem avoids the
`/mnt/c` cross-filesystem performance and permission penalties.

---

## 4. Environment repairs made during P0-A

These were necessary to reach G0 and are recorded for transparency:

1. **Missing base tooling** — the base image had no `git`, `curl`, `gcc`, `make`,
   `pip`, `venv`-adjacent dev headers. Installed via `apt-get`:
   `git curl ca-certificates build-essential unzip xz-utils python3-venv python3-dev`.
2. **Broken apt state (one targeted fix)** — `apt` reported
   `perl-base 5.38.2-3.2ubuntu0.2` installed while the only available `perl`
   candidate was `5.38.2-3.2build2`, because the image was pinned to a single
   snapshot suite (`noble`) without `noble-updates`. Fix: added the matching
   snapshot `noble-updates` pocket
   (`/etc/apt/sources.list.d/noble-updates.list`) for the same snapshot timestamp
   `20260827T000000Z`; the original `/etc/apt/sources.list` was **not** modified.
3. **`uv` installed** via the official installer into `~/.local/bin`.

No GPU driver, kernel, Docker or system-wide service configuration was changed.
No Linux NVIDIA kernel driver was installed.

---

## 5. G0-A CUDA smoke

Performed **after** the P0-B PyTorch install (as specified in the execution order).

```python
import torch
assert torch.cuda.is_available()
x = torch.ones((8, 8), device="cuda")
assert x.sum().item() == 64
```

Measured output:

```
torch 2.9.1+cu128
torch.cuda 12.8
cuda_available True
gpu_name NVIDIA GeForce RTX 4060 Ti
cuda_8x8_sum 64.0
P0_CUDA_8X8_PASS
```

No VRAM stress test and no GPU performance benchmark was run, per P0 scope.

---

## 6. Notes / limitations

- The distro is a generic Ubuntu 24.04 WSL2 environment; it is **not** a PA3 or
  PA4 production environment. Nothing in this document implies production parity.
- Shell invocations from PowerShell must use `wsl.exe -d <distro> -e bash -c ...`;
  `bash -lc` inherits a Windows `HOME` and resets `PATH`, which hid system binaries.
  The recorded tests used an explicit `PATH`.
- Rust is not installed yet; P1/P2 must install the pinned Rust `1.90.0` toolchain
  before building the PA3 engine.
