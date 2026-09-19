# Sandbox repair notes (DSH on Windows)
#
# `scripts/dsh-sandbox-doctor.ps1`  — diagnose, and recreate a missing temp dir
# `scripts/dsh-temp-watch.ps1`      — keep every known dsh-* temp dir alive
#
# Symptom
# -------
#   Error: sandbox mode "workspace-write" is requested but no sandbox backend is
#   usable on this host; refusing to run the command unconfined. ...
#   Runner failure: windows-acl-run: --temp is not an existing directory:
#   C:\Users\<user>\AppData\Local\Temp\dsh-XXXXXXXX
#
# Every command fails this way -- including `Write-Output "hello"` -- so it looks
# like the agent "cannot run the command line" at all.  It can: the ACL runner
# DSH wraps each command in refuses to start.
#
# Cause
# -----
# dsh-sandbox-local creates one private temp directory per (session, workspace)
# with `mkdtempSync(join(tmpdir(), "dsh-"))` and hands the path to the runner as
# `--temp`; the runner validates it exists before doing anything else
# (`requireDirectory` in dsh-sandbox-windows-acl/lib/runner.js).  The path is
# cached in `tempCapabilities` for the provider's lifetime and only recreated on
# `dispose()` (provider shutdown), so if a cleaner, a disk tool or a reboot
# removes that folder, the session is stuck: nothing recreates it.
#
# Fix now (no restart)
# --------------------
#   & .\scripts\dsh-temp-watch.ps1 -Once
# It scans DSH's own session records for the runner's
# "--temp is not an existing directory: <path>" message and re-creates exactly
# those paths.  Verified: after recreating `dsh-rmfK2F`, the very next command
# ran in the ordinary workspace-write mode with no approval prompt.
#
# Fix permanently (needs one DSH restart)
# --------------------------------------
# Point DSH's temp root at a directory nothing cleans, *before* starting DSH:
#
#   mkdir D:\dsh-temp
#   $env:TEMP = 'D:\dsh-temp'; $env:TMP = 'D:\dsh-temp'
#   dsh web
#
# DSH reads the temp root at startup, so the private directory it caches will
# then live outside the auto-cleaned Windows Temp folder.
#
# Belt and braces
# ---------------
#   & .\scripts\dsh-sandbox-doctor.ps1 -InstallWatch
# registers a scheduled task that re-creates any missing dsh-* directory every
# few minutes (needs an elevated shell to register).
#
# Note for future automation: recreating the directory is only half of it.  The
# runner also checks that the path carries the expected write ACE, and an empty
# re-created folder has no such ACE.  It worked here (the ACLs DSH granted to
# the parent folder evidently still cover it), but if a re-created directory is
# rejected, the reliable move is a DSH restart with TEMP pointed at D:\dsh-temp.

# ---- cleanup note -----------------------------------------------------------
# An earlier version of the watcher matched bare `dsh-XXXXXX` tokens in the
# session transcripts and created ~50 bogus empty directories in
# %TEMP%\dsh-<six letters>.  The regex now only accepts the runner's own error
# text.  The leftovers are harmless (empty folders, never referenced by DSH);
# delete them with an elevated shell if they bother you:
#
#   Get-ChildItem $env:TEMP -Directory -Filter 'dsh-*' |
#     Where-Object { $_.Name -notin @('dsh-acl-locks') -and
#                    $_.Name -notlike 'dsh-spill-*' -and
#                    $_.Name -notlike 'dsh-subprocess-*' -and
#                    $_.Name -ne 'dsh-rmfK2F' } |
#     Where-Object { -not (Get-ChildItem $_.FullName -Force) } |
#     Remove-Item -Recurse -Force
