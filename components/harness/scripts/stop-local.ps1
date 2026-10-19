param([switch]$Check)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

function Get-ServiceKind($Process, [string]$Root) {
    # Exact executable/script paths, never just python.exe/node.exe or a port.
    $python = [regex]::Escape((Join-Path $Root '.venv\Scripts\python.exe'))
    $serve = [regex]::Escape((Join-Path $Root '.venv\Scripts\a4s-serve.exe'))
    $vite = [regex]::Escape((Join-Path $Root 'frontend\node_modules\vite\bin\vite.js'))
    $command = [string]$Process.CommandLine
    $pythonPrefix = '^"?' + $python + '"?\s+'
    $module = '-m\s+a4s\.server\.app(?:\s|$)'
    $inline = '-c\s+"from a4s\.server import main; raise SystemExit\(main\(\)\)"(?:\s|$)'
    $consoleScript = '"?' + $serve + '"?(?:\s|$)'
    if ($command -match ('^' + $consoleScript) -or
        $command -match ($pythonPrefix + '(?:' + $module + '|' + $inline + '|' + $consoleScript + ')')) {
        return 'A4S backend'
    }
    if ($Process.Name -eq 'node.exe' -and
        $command -match ('^(?:"[^"]+"|\S+)\s+"?' + $vite + '"?(?:\s|$)')) {
        return 'Vite frontend'
    }
    return $null
}

function Get-Descendants($RootProcess, $Processes) {
    # Take a fixed snapshot. Child creation must be after this particular parent
    # incarnation, preventing a reused parent PID from claiming unrelated work.
    foreach ($child in $Processes) {
        if ($child.ParentProcessId -eq $RootProcess.ProcessId -and
            $child.ProcessId -ne $RootProcess.ProcessId -and
            $child.CreationDate -ge $RootProcess.CreationDate) {
            $child
            Get-Descendants $child $Processes
        }
    }
}

function Stop-VerifiedProcess($Snapshot) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($Snapshot.ProcessId)"
    if (!$current) { return }
    if ($current.CreationDate -ne $Snapshot.CreationDate -or
        $current.CommandLine -cne $Snapshot.CommandLine -or
        $current.ParentProcessId -ne $Snapshot.ParentProcessId) {
        throw "PID $($Snapshot.ProcessId) changed identity; not stopped. Run stop again to reassess."
    }
    # Acquire a process handle and recheck creation time before termination.
    # Never use taskkill /IM or kill every owner of a familiar port.
    $handle = Get-Process -Id $Snapshot.ProcessId -ErrorAction SilentlyContinue
    if (!$handle) { return }
    try {
        $null = $handle.Handle
        if ([math]::Abs(($handle.StartTime.ToUniversalTime() - $Snapshot.CreationDate.ToUniversalTime()).TotalMilliseconds) -gt 1) {
            throw "PID $($Snapshot.ProcessId) was reused; not stopped."
        }
        Stop-Process -InputObject $handle -Force -ErrorAction Stop
        if (!$handle.WaitForExit(5000)) { throw "PID $($Snapshot.ProcessId) did not exit." }
    } catch {
        # A child may exit naturally when its parent stops. That is success,
        # not a reason to fall back to killing a fresh process with the same PID.
        if (!$handle.HasExited) { throw }
    } finally {
        $handle.Dispose()
    }
}

function Stop-LocalServices([switch]$InspectOnly) {
    $processes = @(Get-CimInstance Win32_Process)
    $targets = @($processes | Where-Object { Get-ServiceKind $_ $projectRoot })
    if (!$targets.Count) {
        Write-Host 'No matching A4S backend or Vite frontend process is running.'
    }
    $stopped = @{}
    foreach ($target in $targets) {
        $kind = Get-ServiceKind $target $projectRoot
        $children = @(Get-Descendants $target $processes)
        if ($InspectOnly) {
            Write-Host "[would stop] $kind PID $($target.ProcessId), $($children.Count) child process(es)"
            continue
        }
        # Stop the parent first, then its captured children so it cannot spawn
        # more work during normal teardown. This interrupts active Agent work.
        foreach ($item in @($target) + $children) {
            if (!$stopped.ContainsKey([string]$item.ProcessId)) {
                Stop-VerifiedProcess $item
                $stopped[[string]$item.ProcessId] = $true
            }
        }
        Write-Host "[stopped] $kind PID $($target.ProcessId)"
    }
    if (!$InspectOnly) {
        $remaining = @(Get-CimInstance Win32_Process | Where-Object { Get-ServiceKind $_ $projectRoot })
        if ($remaining.Count) { throw 'A matching service is still running or was restarted; inspect before retrying.' }
    }
    Write-Host 'Docker, databases, files and unrelated application processes are untouched.'
}

# Dot-sourcing exposes only the pure identification helpers for offline tests.
if ($MyInvocation.InvocationName -ne '.') {
    try {
        Stop-LocalServices -InspectOnly:$Check
        exit 0
    } catch {
        Write-Error "Stop failed: $($_.Exception.Message)" -ErrorAction Continue
        exit 1
    }
}
