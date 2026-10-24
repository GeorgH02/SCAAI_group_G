<#
.SYNOPSIS
Start the local Vienna Stay llama.cpp server. Stop with Ctrl+C.
.EXAMPLE
.\scripts\start_llm.ps1
.EXAMPLE
.\scripts\start_llm.ps1 -ModelFile 'D:\Models\qwen.gguf' -GpuLayers 0
#>
[CmdletBinding()]
param(
    [string]$Server = 'llama-server',
    [string]$ModelRepo = 'Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M',
    [string]$ModelFile,
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [ValidateRange(512, 32768)][int]$ContextSize = 16384,
    [ValidatePattern('^(auto|all|[0-9]+)$')][string]$GpuLayers = 'auto',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$serverArguments = @('--host', '127.0.0.1', '--port', "$Port",
    '--alias', 'vienna-stay', '--ctx-size', "$ContextSize", '--parallel', '1',
    '--jinja', '--gpu-layers', $GpuLayers, '--log-verbosity', '2')
if ($ModelFile) {
    if (-not (Test-Path -LiteralPath $ModelFile -PathType Leaf)) {
        throw "Model file not found: $ModelFile"
    }
    $serverArguments += @('--model', (Resolve-Path -LiteralPath $ModelFile).Path)
} else {
    if ([string]::IsNullOrWhiteSpace($ModelRepo) -or $ModelRepo.StartsWith('-')) {
        throw 'ModelRepo must specify a Hugging Face GGUF repository and quantization.'
    }
    $serverArguments += @('--hf-repo', $ModelRepo)
}

if ($DryRun) {
    # JSON preserves argument boundaries, including paths containing spaces.
    ConvertTo-Json -InputObject (@($Server) + $serverArguments)
    return
}
$executable = Get-Command $Server -CommandType Application -ErrorAction SilentlyContinue
if (-not $executable) {
    throw 'llama-server not found. Run: winget install llama.cpp; then reopen PowerShell. Or pass -Server with the full executable path.'
}
Write-Host "Starting local LLM at http://127.0.0.1:$Port (alias: vienna-stay)."
Write-Host 'The first repository-based launch downloads model weights. Stop with Ctrl+C.'
& $executable.Source @serverArguments
exit $LASTEXITCODE
