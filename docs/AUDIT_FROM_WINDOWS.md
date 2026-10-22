# Auditing PA4 from Windows (bypassing WSL networking)

**Status:** established procedure, used to close P3.5
**Applies to:** any fresh-context `pa4-auditor` review of the PA4 repository

---

## 1. Why this exists

PA4's periodic audits need an **independent, read-only, fresh-context** reviewer
that can read the **actual repository source**, not a summary. Two environments
make that hard:

| Problem | Symptom |
|---|---|
| **WSL network instability** | `ConnectionRefused` / TCP timeouts while the WSL session runs long agent turns. Root cause measured on 2026-10-22: OpenCode opens a new TCP connection per API request, so ephemeral ports exhaust. Windows reported 10/10 success to the same endpoint while WSL failed 13/20. |
| **Auditor sandbox boundaries** | An auditor spawned from a session rooted at `C:\Users\임민우\Desktop\PA4` inherits that root and is denied `external_directory` access to `/home/pa4/...`. It then reads **zero files** and cannot audit. |

Running the reviewer **from Windows against an exported copy** removes both
problems. The export is a byte-exact `git archive`, so auditing it is auditing
that commit.

**This is not a weaker audit.** It is the same source bytes, the same tests, and
the same PA3 reference — just reachable without WSL's network stack.

---

## 2. Procedure

### Step 1 — Record the tree you are about to audit

```bash
wsl -e bash -c 'cd ~/projects/PokeAgent4.0 && git rev-parse HEAD && git status --porcelain'
```

The worktree must be **clean**. An export of a dirty tree silently omits
uncommitted changes, which would make the audit unsound.

### Step 2 — Export

Run from PowerShell:

```powershell
scripts\export_audit.ps1 <sha>
```

Or manually:

```powershell
$ref  = "ecfe72e"
$dest = "C:\Users\임민우\Desktop\PA4\audit_$ref"
wsl -e bash -c "rm -rf '$dest'; mkdir -p '$dest'; cd ~/projects/PokeAgent4.0 && git archive $ref | tar -x -C '$dest'"
wsl -e bash -c "cp -r /home/pa4/projects/_pa3_ref/agent '$dest/_pa3_ref_agent'"
wsl -e bash -c "cp /home/pa4/.pa3_build/engine/python/pa3_engine/observation.py '$dest/_pa3_ref_agent/observation_pa3.py'"
```

The PA3 copy matters: several PA4 contracts (ragged layout, observation schema)
are defined by PA3's source, and an auditor that cannot read it can only check
PA4's claims indirectly.

### Step 3 — Prove access before asking for a verdict

**Always** require the auditor's first action to be a literal file read:

> Read `<dest>\pa4\env\ragged_transport.py` and quote its first 5 lines verbatim.

If it reports `permission.rejected` or `external_directory`, **stop**. Do not
substitute an inline review and do not accept a verdict. That exact failure was
mistaken for a PASS once (P3, G3-E).

### Step 4 — Audit with a falsification prompt

Ask for **specific, attackable** claims, not "review this". Good questions name
the failure mode:

- Can the mechanism actually **fail**, or is it constructed to always pass?
- Does the test traverse the **real** seam, or a shortcut?
- Is the evidence **reproducible from committed code**, or only asserted?
- Is any claimed number **absent** from an evidence file?

### Step 5 — Re-verify the auditor's findings yourself

An auditor is a process control, not an oracle. Before acting, reproduce each
MAJOR finding against the real tree. In P3.5 the external audit's four MAJOR
findings were all confirmed by direct experiment before any fix was accepted.

### Step 6 — Record, then close

Update the phase receipt with:

- `auditor_read_real_repo: true`
- the auditor's verdict and tree
- the access method (this document)
- the lead's own independent verification

Only then flip a gate. **Never mark an audit gate PASS on a review that could
not read the files.**

---

## 3. Interpreting verdicts

| Verdict | Meaning | Action |
|---|---|---|
| `CLEAN` | No findings | Close the gate |
| `FINDINGS (MINOR)` | No behavioural defect; documentation/evidence gaps | Fix the docs, close the gate, record the nits |
| `FINDINGS (MAJOR)` | A real defect | Fix, then re-audit once |
| `INSUFFICIENT_EVIDENCE` | Could not read the target | **Not a pass and not a rejection.** Fix access and retry |

---

## 4. Stopping rule

An audit loop must converge. **Two clean rounds is the signal to proceed.**

If a third round produces only MINOR findings, stop: closing documentation nits
is not worth another full audit cycle. P3.5 reached round ten on minors before
being stopped — that is the failure mode this rule prevents.

Record remaining MINOR items as **documented limitations**, not blockers.

---

## 5. What this procedure does NOT cover

- **Executing the test suite.** The auditor is read-only. Objective test results
  must come from the lead, committed as evidence (e.g. `evidence/full_suite.txt`).
  An audit without a committed test log leaves a real evidence gap.
- **Proving the export matches the live tree.** `git archive` of a clean tree at
  a recorded SHA does; an export of a dirty tree does not.
- **PA3 write access.** PA3 stays read-only at its pinned SHA. The copy is for
  reading only.

---

## 6. Files

| Path | Purpose |
|---|---|
| `docs/AUDIT_FROM_WINDOWS.md` | this procedure |
| `scripts/export_audit.ps1` | one-command export helper |
| `docs/phases/P3.5/evidence/` | worked example: the P3.5 external audit |
