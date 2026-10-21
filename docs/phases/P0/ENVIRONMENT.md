# P0-A — Development Host Environment

Document ID: `PA4-P0-ENV-v1.1`
Scope: P0-A (host/WSL2/GPU preflight). Observed on the actual development workstation.

> **Revision note (v1.1):** added raw `uname`/`/proc/version`/`os-release` output proving the
> WSL2 host identity, which v1.0 asserted without inline evidence.

---

## 1. Host identity (measured, same session as the passing smoke)

```
$ uname -a
Linux DESKTOP-CV81OHM 6.18.35.2-microsoft-standard-WSL2+ #1 SMP PREEMPT_DYNAMIC Tue Sep  1 14:11:36 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux

$ cat /proc/version
Linux version 6.18.35.2-microsoft-standard-WSL2+ (runner@runnervmgx7h7) (gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, GNU ld (GNU Binutils for Ubuntu) 2.42) #1 SMP PREEMPT_DYNAMIC Tue Sep  1 14:11:36 UTC 2026

$ grep -E "^(NAME|VERSION_ID)=" /etc/os-release
NAME="Ubuntu"
VERSION_ID="24.04"
```

The `microsoft-standard-WSL2+` kernel tag is the WSL2 proof.

| Item | Measured |
|---|---|
| Host OS | Windows 11 (`wsl --version` reports Windows `10.0.26300.9711`) |
| WSL version | `2.7.12.0`; kernel `6.18.33.2-2`; WSLg `1.0.73.2` |
| Distro | Ubuntu `24.04.3 LTS (Noble Numbat)`, `VERSION_ID="24.04"` |
| Hostname | `DESKTOP-CV81OHM` |
| NVIDIA driver (Windows) | `610.88`, CUDA UMD `13.3` |
| GPU | `NVIDIA GeForce RTX 4060 Ti` (8 GB / `8188 MiB`) |
| GPU in WSL | `nvidia-smi` → `NVIDIA GeForce RTX 4060 Ti, 610.88, 8188 MiB` |

GPU is exposed via the standard Windows WSL passthrough. **No Linux NVIDIA kernel driver was
installed inside WSL.**

---

## 2. Toolchains (inside WSL2)

| Tool | Version | Notes |
|---|---|---|
| Python | `3.12.3` | `/usr/bin/python3.12` |
| Git | `2.43.0` | installed during P0-A (absent in base image) |
| gcc / make | `13.3.0` / `4.3` | build-essential, for native wheels |
| curl | `8.5.0` | installed during P0-A |
| uv | `0.13.4` | `~/.local/bin` |
| Rust / Cargo | **not installed** | PA3 pins Rust `1.90.0`; needed in P1/P2 |

---

## 3. Workspace

| Item | Measured |
|---|---|
| PA4 repo | `/root/projects/PokeAgent4.0` (WSL ext4, **not** `/mnt/c`) |
| Root filesystem | `/dev/sdc`, 1007G total, ~956G free |
| PA3 reference | `/root/projects/_pa3_ref/pa3` (separate directory) |
| Windows working copy | `C:\Users\<user>\Desktop\PA4` (source design docs only) |

---

## 4. Environment repairs made during P0-A

1. **Missing base tooling** — no `git`, `curl`, `gcc`, `make`, `pip`. Installed via `apt-get`:
   `git curl ca-certificates build-essential unzip xz-utils python3-venv python3-dev`.
2. **Broken apt state (one targeted fix)** — `perl-base 5.38.2-3.2ubuntu0.2` was installed
   while the only `perl` candidate was `5.38.2-3.2build2`, because the image was pinned to a
   single snapshot suite (`noble`) lacking `noble-updates`. Fix: added the matching snapshot
   `noble-updates` pocket at the same timestamp (`20260827T000000Z`) in
   `/etc/apt/sources.list.d/noble-updates.list`. `/etc/apt/sources.list` was **not** modified.
3. **`uv` installed** via the official installer into `~/.local/bin`.

No GPU driver, kernel, Docker or system service configuration was changed.

---

## 5. G0-A CUDA smoke

Performed **after** the P0-B PyTorch install, as specified by the execution order.

```
torch 2.9.1+cu128
torch.cuda 12.8
cuda_available True
gpu_name NVIDIA GeForce RTX 4060 Ti
cuda_8x8_sum 64.0
P0_CUDA_8X8_PASS
```

The check is `ones((8,8), device='cuda').sum() == 64.0` plus a device-type assertion.
No VRAM stress test and no GPU benchmark was run, per P0 scope.

Note: `torch.cuda` (`12.8`) is the CUDA version the wheel was **built** against; it is not
by itself proof of driver capability. Driver health is evidenced separately by `nvidia-smi`
reporting `610.88` inside WSL.

---

## 6. Notes / limitations

- This is a generic Ubuntu 24.04 WSL2 environment, not a PA3/PA4 production environment.
  Nothing here implies production parity.
- Shell invocation must use `wsl.exe -d <distro> -e bash -c ...`; `bash -lc` inherits a Windows
  `HOME` and resets `PATH`, which hid system binaries during preflight.
- Rust `1.90.0` must be installed before P1/P2 can build `pa3-engine`.
