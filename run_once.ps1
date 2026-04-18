Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv\Scripts\python.exe. The local runtime has not been prepared."
}

$configPath = Join-Path $projectRoot "config.py"
$envPath = Join-Path $projectRoot ".env"
$hasApiKeyInEnv = [bool]$env:OPENROUTER_API_KEY
$hasConfigFile = Test-Path $configPath
$hasEnvFile = Test-Path $envPath

if (-not ($hasApiKeyInEnv -or $hasConfigFile -or $hasEnvFile)) {
    throw "Missing OpenRouter configuration. Add your key to config.py, .env, or OPENROUTER_API_KEY before running."
}

$tmpDir = Join-Path $projectRoot ".tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$env:TEMP = $tmpDir
$env:TMP = $tmpDir

if ($hasConfigFile) {
    $configText = Get-Content $configPath -Raw
    if ($configText -match 'your-openrouter-api-key|your-api-key-here|YOUR_API_KEY') {
        throw "config.py still has the placeholder API key. Replace it with your real OpenRouter key and run again."
    }
}

& $python ".\data_engineering\pipeline\run_evaluation.py" `
    --samples 1 `
    --strategies baseline,iter_refine,filter,majority `
    --max-iterations 1 `
    --filter-samples 2 `
    --majority-samples 3
