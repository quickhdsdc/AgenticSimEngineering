"""Windows stop helper checks. Never stop actual project services in tests."""

import base64
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/stop-local.ps1"


def run_check(body):
    source = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n. '" + str(SCRIPT).replace("'", "''") + "'\n" + body
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stderr


def test_exact_project_paths_and_service_commands():
    run_check(r'''
$root = 'C:\Example Project\harness'
function Assert-Kind($name, $command, $expected) {
    $actual = Get-ServiceKind ([pscustomobject]@{Name=$name; CommandLine=$command}) $root
    if ($actual -ne $expected) { throw "Unexpected service classification: $actual" }
}
Assert-Kind 'python.exe' '"C:\Example Project\harness\.venv\Scripts\python.exe" -m a4s.server.app --port 8000' 'A4S backend'
Assert-Kind 'python.exe' '"C:\Example Project\harness\.venv\Scripts\python.exe" -c "from a4s.server import main; raise SystemExit(main())" --port 8000' 'A4S backend'
Assert-Kind 'python.exe' '"C:\Example Project\harness\.venv\Scripts\python.exe" "C:\Example Project\harness\.venv\Scripts\a4s-serve.exe"' 'A4S backend'
Assert-Kind 'a4s-serve.exe' '"C:\Example Project\harness\.venv\Scripts\a4s-serve.exe" --port 8000' 'A4S backend'
Assert-Kind 'node.exe' '"C:\Program Files\nodejs\node.exe" "C:\Example Project\harness\frontend\node_modules\vite\bin\vite.js" --port 5173' 'Vite frontend'
Assert-Kind 'python.exe' '"C:\Other\.venv\Scripts\python.exe" -m a4s.server.app --port 8000' $null
Assert-Kind 'python.exe' '"C:\Example Project\harness\.venv\Scripts\python.exe" -m a4s.cli run task' $null
Assert-Kind 'python.exe' '"C:\Example Project\harness\.venv\Scripts\python.exe" -c "print(''a4s.server.app'')"' $null
Assert-Kind 'node.exe' 'node.exe C:\Other\frontend\node_modules\vite\bin\vite.js --port 5173' $null
Assert-Kind 'node.exe' 'node.exe "C:\Example Project\harness\frontend\node_modules\vite\bin\vite.js.other"' $null
Assert-Kind 'node.exe' 'node.exe --eval "console.log(''C:\Example Project\harness\frontend\node_modules\vite\bin\vite.js'')"' $null
Assert-Kind 'python.exe' '' $null
''')


def test_descendants_exclude_stale_parent_pid_and_unrelated_processes():
    run_check(r'''
$root = [pscustomobject]@{ProcessId=10; ParentProcessId=1; CreationDate=[datetime]'2026-09-13T10:00:00'}
$child = [pscustomobject]@{ProcessId=11; ParentProcessId=10; CreationDate=[datetime]'2026-09-13T10:01:00'}
$grandchild = [pscustomobject]@{ProcessId=12; ParentProcessId=11; CreationDate=[datetime]'2026-09-13T10:02:00'}
$stale = [pscustomobject]@{ProcessId=13; ParentProcessId=10; CreationDate=[datetime]'2026-09-12T10:01:00'}
$other = [pscustomobject]@{ProcessId=14; ParentProcessId=90; CreationDate=[datetime]'2026-09-13T10:01:00'}
$children = @(Get-Descendants $root @($root,$child,$grandchild,$stale,$other))
if (($children.ProcessId -join ',') -ne '11,12') { throw 'Wrong process tree' }
''')


def test_pid_identity_change_refuses_termination():
    run_check(r'''
$snapshot = [pscustomobject]@{ProcessId=123; ParentProcessId=1; CommandLine='old'; CreationDate=[datetime]'2026-09-13T10:00:00'}
function Get-CimInstance { [pscustomobject]@{ProcessId=123; ParentProcessId=1; CommandLine='new'; CreationDate=[datetime]'2026-09-13T10:01:00'} }
function Stop-Process { throw 'UNSAFE_STOP' }
$refused = $false
try { Stop-VerifiedProcess $snapshot } catch {
    if ($_.Exception.Message -notlike '*changed identity*') { throw }
    $refused = $true
}
if (!$refused) { throw 'Did not reject reused PID' }
''')


@pytest.mark.integration
def test_stop_verified_disposable_child_only():
    # Own throwaway child: no backend, browser, Docker or model involved.
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        run_check(f'''
$snapshot = Get-CimInstance Win32_Process -Filter 'ProcessId = {process.pid}'
if (!$snapshot) {{ throw 'Test child missing' }}
Stop-VerifiedProcess $snapshot
Stop-VerifiedProcess $snapshot
''')
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
