# Keep the DSH sandbox's private temp directory alive.
#
# DSH hands the windows-acl runner a private temp directory (--temp) that it
# created once per session with mkdtempSync(join(tmpdir(), "dsh-")), and the
# runner refuses to start unless that directory exists.  The path is cached for
# the provider's lifetime, so if a cleaner removes it, every later command fails
# with:
#
#   Runner failure: windows-acl-run: --temp is not an existing directory: <path>
#
# This watcher cannot know the random name of the *next* directory DSH will
# create, but it can remember every dsh-* directory that has ever existed and
# re-create the empty ones, which is exactly what the runner checks for.  It is
# deliberately conservative: it only creates missing directories, never deletes
# or touches anything that exists.
#
# Run it once:      pwsh -File scripts\dsh-temp-watch.ps1 -Once
# Run it in a loop: pwsh -File scripts\dsh-temp-watch.ps1 -IntervalSeconds 60
#
# Install it as a scheduled task with:
#   pwsh -File scripts\dsh-sandbox-doctor.ps1 -InstallWatch

[CmdletBinding()]
param(
    [string]$TempRoot = [System.IO.Path]::GetTempPath(),
    [int]$IntervalSeconds = 0,
    [int]$KeepLast = 20,
    [switch]$Once,
    [string]$LogPath
)

$ErrorActionPreference = 'Continue'
$statePath = Join-Path $env:USERPROFILE '.dsh-temp-watch-state.json'
if (-not $LogPath) { $LogPath = Join-Path $env:USERPROFILE '.dsh-temp-watch.log' }

function Write-Log([string]$message) {
    $line = '{0}  {1}' -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), $message
    Write-Host $line
    try { Add-Content -LiteralPath $LogPath -Value $line -ErrorAction Stop } catch { }
}

function Get-SeenPaths {
    if (Test-Path -LiteralPath $statePath) {
        try { return @((Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json)) } catch { }
    }
    return @()
}

function Get-MissingPathFromSessions {
    # The directory name is random per session and the error text is returned to
    # the caller rather than to this process, so recover it from DSH's own
    # session records: the failure is one of the last things written.
    $sessions = Join-Path $env:USERPROFILE '.dsh\sessions'
    if (-not (Test-Path -LiteralPath $sessions)) { return @() }
    $found = @()
    foreach ($file in Get-ChildItem -LiteralPath $sessions -Recurse -File -ErrorAction SilentlyContinue |
                      Sort-Object LastWriteTime -Descending | Select-Object -First 3) {
        $text = $null
        if ($file.Extension -eq '.zstd') {
            $zstd = Get-Command zstd.exe -ErrorAction SilentlyContinue
            if (-not $zstd) { continue }
            $tmp = Join-Path $env:USERPROFILE ('.dsh-scan-' + [guid]::NewGuid().ToString('N').Substring(0, 6) + '.jsonl')
            try {
                & $zstd.Source -d -f -q -o $tmp $file.FullName 2>$null | Out-Null
                if (Test-Path -LiteralPath $tmp) { $text = Get-Content -LiteralPath $tmp -Raw }
            } catch { } finally { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
        } else {
            try { $text = Get-Content -LiteralPath $file.FullName -Raw } catch { }
        }
        if ($text) {
            # Only trust the runner's own error text: "--temp is not an existing
            # directory: <path>".  Matching bare "dsh-XXXXXX" tokens instead
            # dredges up arbitrary six-character words from the transcripts and
            # creates dozens of bogus directories.
            foreach ($m in [regex]::Matches($text, 'is not an existing directory:\s*([A-Za-z]:\\[^"\\]+(?:\\[^"\\]+)*)')) {
                $found += $m.Groups[1].Value.Trim()
            }
        }
    }
    return @($found | Sort-Object -Unique)
}

function Save-SeenPaths($paths) {
    try { ($paths | Select-Object -Last $KeepLast) | ConvertTo-Json | Set-Content -LiteralPath $statePath } catch { }
}

function Repair-Once {
    if (-not (Test-Path -LiteralPath $TempRoot)) {
        Write-Log "temp root '$TempRoot' is missing; creating it"
        try { New-Item -ItemType Directory -Path $TempRoot -Force -ErrorAction Stop | Out-Null }
        catch { Write-Log "  failed: $($_.Exception.Message)" }
    }

    $seen = @(Get-SeenPaths)
    $live = @()
    if (Test-Path -LiteralPath $TempRoot) {
        $live = @(Get-ChildItem -LiteralPath $TempRoot -Directory -Filter 'dsh-*' -ErrorAction SilentlyContinue |
                  Select-Object -ExpandProperty FullName)
    }
    $fromSessions = @(Get-MissingPathFromSessions)
    if ($fromSessions.Count -gt 0) { Write-Log "candidates from DSH session records: $($fromSessions.Count)" }

    # remember anything currently present, so it can be re-created if it vanishes
    $all = @($seen + $live + $fromSessions | Sort-Object -Unique)
    $repaired = 0
    foreach ($path in $all) {
        if (-not (Test-Path -LiteralPath $path)) {
            try {
                New-Item -ItemType Directory -Path $path -Force -ErrorAction Stop | Out-Null
                Write-Log "re-created $path"
                $repaired++
            } catch {
                Write-Log "could not re-create $path : $($_.Exception.Message)"
            }
        }
    }
    Save-SeenPaths $all
    if ($repaired -eq 0) { Write-Log "ok: $($all.Count) known dsh temp dir(s), none missing" }
    return $repaired
}

if ($Once -or $IntervalSeconds -le 0) {
    Write-Log "dsh-temp-watch: tempRoot='$TempRoot' state='$statePath'"
    [void](Repair-Once)
    exit 0
}

Write-Log "dsh-temp-watch: watching every ${IntervalSeconds}s (tempRoot='$TempRoot')"
while ($true) {
    [void](Repair-Once)
    Start-Sleep -Seconds $IntervalSeconds
}
