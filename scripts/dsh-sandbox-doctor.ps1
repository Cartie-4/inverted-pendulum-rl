# Repair / diagnose the DSH Windows sandbox runner.
#
# Symptom it fixes:
#   Error: sandbox mode "workspace-write" is requested but no sandbox backend is
#   usable on this host ... Runner failure: windows-acl-run: --temp is not an
#   existing directory: C:\Users\<user>\AppData\Local\Temp\dsh-XXXXXXXX
#
# Cause: sandbox-local materialises one private temp directory per (session,
# workspace) with mkdtempSync(join(tmpdir(), "dsh-")) and hands it to the runner
# as --temp; the runner refuses to start unless that directory exists.  The path
# is cached for the provider's lifetime, so if something (a cleaner, a disk tool,
# a reboot of the temp folder) removes it, every later command fails with the
# error above and nothing recreates it.
#
# What this script does:
#   1. reports the temp root, the DSH directories it can see, and whether they
#      are writable;
#   2. recreates the exact missing path when it is given one;
#   3. optionally installs a scheduled task that re-creates every *cached* dsh-*
#      directory that DSH later complains about -- see dsh-temp-watch.ps1.
#
# Usage:
#   pwsh -File scripts\dsh-sandbox-doctor.ps1
#   pwsh -File scripts\dsh-sandbox-doctor.ps1 -MissingPath 'C:\Users\me\AppData\Local\Temp\dsh-abc123'
#   pwsh -File scripts\dsh-sandbox-doctor.ps1 -InstallWatch

[CmdletBinding()]
param(
    [string]$MissingPath,
    [switch]$InstallWatch,
    [int]$IntervalMinutes = 5
)

$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$watchScript = Join-Path $here 'dsh-temp-watch.ps1'
$taskName = 'DSH-Sandbox-Temp-Repair'

function Write-Head($text) { Write-Host ""; Write-Host "=== $text ===" -ForegroundColor Cyan }

Write-Head '1. temp roots'
foreach ($name in 'TEMP', 'TMP') {
    $value = [Environment]::GetEnvironmentVariable($name)
    $exists = if ($value) { Test-Path -LiteralPath $value } else { $false }
    Write-Host ("  {0,-5} = {1,-60} exists={2}" -f $name, $value, $exists)
}
$tempRoot = [System.IO.Path]::GetTempPath()
Write-Host ("  GetTempPath() = {0}" -f $tempRoot)

Write-Head '2. dsh-* directories under the temp root'
$candidates = @()
if (Test-Path -LiteralPath $tempRoot) {
    $candidates = Get-ChildItem -LiteralPath $tempRoot -Directory -Filter 'dsh-*' -ErrorAction SilentlyContinue
    if ($candidates) {
        foreach ($c in $candidates) {
            $writable = $false
            try {
                $probe = Join-Path $c.FullName ('.write-probe-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
                Set-Content -LiteralPath $probe -Value 'ok' -ErrorAction Stop
                Remove-Item -LiteralPath $probe -Force -ErrorAction Stop
                $writable = $true
            } catch { $writable = $false }
            Write-Host ("  {0}  lastWrite={1:yyyy-MM-dd HH:mm:ss}  writable={2}" -f $c.Name, $c.LastWriteTime, $writable)
        }
    } else {
        Write-Host '  none found (a running DSH session normally has one)'
    }
} else {
    Write-Host ("  temp root {0} does not exist" -f $tempRoot) -ForegroundColor Yellow
}

Write-Head '3. recreate a missing path'
if ($MissingPath) {
    if (Test-Path -LiteralPath $MissingPath) {
        Write-Host ("  {0} already exists" -f $MissingPath)
    } else {
        try {
            New-Item -ItemType Directory -Path $MissingPath -Force -ErrorAction Stop | Out-Null
            Write-Host ("  created {0}" -f $MissingPath) -ForegroundColor Green
            Write-Host '  now retry the command that failed.'
        } catch {
            Write-Host ("  could not create {0}: {1}" -f $MissingPath, $_.Exception.Message) -ForegroundColor Red
        }
    }
} else {
    Write-Host '  (pass -MissingPath "<the path DSH printed>" to recreate it; the name is random per session)'
}

Write-Head '4. optional watch task'
if ($InstallWatch) {
    $action = New-ScheduledTaskAction -Execute 'pwsh' `
        -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -File "{0}" -TempRoot "{1}"' -f $watchScript, $tempRoot)
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    try {
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force -ErrorAction Stop | Out-Null
        Write-Host ("  installed scheduled task '{0}' every {1} min" -f $taskName, $IntervalMinutes) -ForegroundColor Green
    } catch {
        Write-Host ("  could not register the task: {0}" -f $_.Exception.Message) -ForegroundColor Red
        Write-Host '  (registering a task usually needs an elevated shell)'
        Write-Host ("  manual alternative: pwsh -File `"{0}`" -TempRoot `"{1}`" in a loop" -f $watchScript, $tempRoot)
    }
} else {
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) { Write-Host ("  task '{0}' is registered (state {1})" -f $taskName, $existing.State) }
    else { Write-Host '  (pass -InstallWatch to register the repair task; needs an elevated shell)' }
}

Write-Head '5. the durable fix (needs a DSH restart, do it yourself)'
Write-Host '  The private temp directory is chosen by DSH at startup from TEMP/TMP and'
Write-Host '  cached for the session.  To keep it out of a folder that gets cleaned:'
Write-Host ''
Write-Host '    mkdir D:\dsh-temp'
Write-Host '    $env:TEMP = "D:\dsh-temp"; $env:TMP = "D:\dsh-temp"'
Write-Host '    dsh web        # start DSH from this shell'
Write-Host ''
Write-Host '  Then run this script again: the reported temp root should be D:\dsh-temp.'
Write-Host '  Until DSH is restarted, the current session keeps using the old (missing) path,'
Write-Host '  so -MissingPath / the watch task is what unblocks it in the meantime.'
Write-Host ''
