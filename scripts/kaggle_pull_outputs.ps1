param(
    [Parameter(Mandatory = $true)]
    [string]$KernelRef,
    [string]$OutputDir = "kaggle/modelisation/outputs",
    [string]$FilePattern = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "Required command not found: kaggle"
}

if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$outputArgs = @("kernels", "output", $KernelRef, "-p", $OutputDir)
if ($Force) {
    $outputArgs += "-o"
}
if (-not [string]::IsNullOrWhiteSpace($FilePattern)) {
    $outputArgs += @("--file-pattern", $FilePattern)
}

Write-Host "Running: kaggle $($outputArgs -join ' ')"
& kaggle @outputArgs

if ($LASTEXITCODE -ne 0) {
    throw "Kaggle output download failed."
}
