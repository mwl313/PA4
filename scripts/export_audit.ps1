# Export the PA4 repository at a commit for read-only auditing from Windows.
#
# Why: WSL networking can intermittently fail (ephemeral port exhaustion), and an
# auditor spawned from a Windows-rooted session cannot read WSL paths. Exporting
# a byte-exact git archive to a Windows-readable directory removes both problems.
#
# Usage (PowerShell):
#   .\export_audit.ps1              # exports HEAD
#   .\export_audit.ps1 ecfe72e      # exports a specific commit
#
# See docs/AUDIT_FROM_WINDOWS.md for the full procedure.

param(
    [string]$Ref = "HEAD"
)

$repo    = "~/projects/PokeAgent4.0"
$winBase = "C:\Users\임민우\Desktop\PA4"
# Inside `wsl bash`, Windows paths must be converted to /mnt/<drive>/... form.
$wslBase = "/mnt/c/Users/임민우/Desktop/PA4"

$resolved = (wsl -e bash -c "cd $repo && git rev-parse --short $Ref").Trim()
if (-not $resolved -or $LASTEXITCODE -ne 0) {
    Write-Error "Could not resolve ref '$Ref' in $repo"
    exit 1
}

$dirty = (wsl -e bash -c "cd $repo && git status --porcelain | wc -l").Trim()
if ($Ref -eq "HEAD" -and $dirty -ne "0") {
    Write-Warning "Worktree has $dirty uncommitted change(s)."
    Write-Warning "git archive exports the COMMIT, not the working tree."
    Write-Warning "Commit or stash before auditing HEAD, or audit a specific SHA."
}

$destW = Join-Path $winBase "audit_$resolved"
$destL = "$wslBase/audit_$resolved"
Write-Host "Exporting $resolved -> $destW"

wsl -e bash -c "rm -rf '$destL'; mkdir -p '$destL'; cd $repo && git archive $resolved | tar -x -C '$destL'"
if ($LASTEXITCODE -ne 0) { Write-Error "export failed"; exit 1 }

# PA3 reference source: several PA4 contracts are defined by PA3's code, and an
# auditor that cannot read it can only check PA4's claims indirectly.
wsl -e bash -c "cp -r /home/pa4/projects/_pa3_ref/agent '$destL/_pa3_ref_agent' 2>/dev/null"
wsl -e bash -c "cp /home/pa4/.pa3_build/engine/python/pa3_engine/observation.py '$destL/_pa3_ref_agent/observation_pa3.py' 2>/dev/null"

# Sanity: the bundle must contain what an audit depends on.
$required = @(
    "pa4\env\ragged_transport.py",
    "pa4\collector\run_batched.py",
    "docs\phases\P3.5\RECEIPT.json"
)
foreach ($f in $required) {
    if (-not (Test-Path (Join-Path $destW $f))) {
        Write-Warning "expected file missing from export: $f"
    }
}

$pyCount = (Get-ChildItem -Path $destW -Recurse -Filter "*.py" -ErrorAction SilentlyContinue).Count
Write-Host ""
Write-Host "Export complete."
Write-Host "  tree  : $resolved"
Write-Host "  path  : $destW"
Write-Host "  files : $pyCount python files"
Write-Host ""
Write-Host "Next: have the auditor read <path>\pa4\env\ragged_transport.py first"
Write-Host "and quote its first 5 lines. If it reports permission.rejected or"
Write-Host "external_directory, STOP - do not accept a verdict."
