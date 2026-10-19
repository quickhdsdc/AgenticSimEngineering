param([ValidateSet('core','dev','harness','workbench','des','newton','identification','semantic','factoryflow','plaid','chrono','fem','isaac','mosim','quarter-car')][string]$Profile = 'workbench', [switch]$Offline)
$ErrorActionPreference = 'Stop'
$networkArgs = @()
if ($Offline) { $networkArgs += '--offline' }
$projectRoot = Split-Path -Parent $PSScriptRoot
$componentRoots = @{
    'des' = 'components/des'
    'isaac' = 'components/dynamics/projects/isaac_sim'
    'mosim' = 'components/dynamics/projects/mosim'
    'quarter-car' = 'components/dynamics/projects/quarter_car_pinn'
}
if ($Profile -eq 'chrono') {
    if ($Offline) { throw 'Chrono bootstrap currently requires public package channels; use the existing installed environment offline.' }
    & (Join-Path $projectRoot '.venv/Scripts/python.exe') (Join-Path $projectRoot 'scripts/bootstrap_chrono.py')
} elseif ($Profile -in @('newton', 'identification', 'semantic', 'factoryflow', 'plaid', 'fem')) {
    $newtonEnv = Join-Path $projectRoot ".venvs/$Profile"
    $newtonPython = Join-Path $newtonEnv 'Scripts/python.exe'
    $cachePath = Join-Path $projectRoot 'work/cache/uv'
    if (-not (Test-Path -LiteralPath $newtonPython)) {
        uv @networkArgs --cache-dir $cachePath venv $newtonEnv --python 3.12
        if ($LASTEXITCODE -ne 0) { throw "$Profile environment creation failed" }
    }
    uv @networkArgs --cache-dir $cachePath pip sync --python $newtonPython (Join-Path $projectRoot "environments/$Profile/requirements.txt")
} elseif ($Profile -eq 'core') {
    uv @networkArgs sync --directory $projectRoot
} elseif ($Profile -eq 'dev') {
    uv @networkArgs sync --locked --directory $projectRoot --extra dev --extra harness --extra des-reference
} elseif ($Profile -eq 'harness') {
    uv @networkArgs sync --locked --directory $projectRoot --extra harness
} elseif ($Profile -eq 'workbench') {
    uv @networkArgs sync --directory $projectRoot --extra workbench
} else {
    uv @networkArgs sync --directory (Join-Path $projectRoot $componentRoots[$Profile])
}
if ($LASTEXITCODE -ne 0) { throw "Environment setup failed: $Profile" }
