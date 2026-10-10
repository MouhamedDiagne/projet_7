param(
    [Parameter(Mandatory = $true)]
    [string]$KernelRef,
    [int]$PollSeconds = 30,
    [switch]$Watch
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "Required command not found: kaggle"
}

do {
    & kaggle kernels status $KernelRef

    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle status check failed."
    }

    if ($Watch) {
        Start-Sleep -Seconds $PollSeconds
    }
}
while ($Watch)
