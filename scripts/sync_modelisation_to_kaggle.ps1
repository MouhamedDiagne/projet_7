param(
    [string]$NotebookPath = "notebooks/modelisation.ipynb",
    [string]$BundlePath = "kaggle/modelisation",
    [string]$MetadataPath = "kaggle/modelisation/kernel-metadata.json",
    [string]$Accelerator = "",
    [switch]$SkipPush
)

$ErrorActionPreference = "Stop"

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

function ConvertTo-AsciiJson {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    # The Kaggle CLI can use the active Windows code page when reading files.
    # Escaping all non-ASCII JSON characters keeps the bundle portable while
    # preserving the original Unicode strings after JSON parsing on Kaggle.
    return [System.Text.RegularExpressions.Regex]::Replace(
        $Content,
        '[^\x00-\x7F]',
        {
            param($match)
            return ('\u{0:x4}' -f [int][char]$match.Value)
        }
    )
}

function Ensure-NotebookKernelspec {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $notebook = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($null -eq $notebook.metadata) {
        $notebook | Add-Member -MemberType NoteProperty -Name metadata -Value ([pscustomobject]@{})
    }
    if ($null -eq $notebook.metadata.kernelspec) {
        $notebook.metadata | Add-Member -MemberType NoteProperty -Name kernelspec -Value ([pscustomobject]@{
            display_name = "Python 3"
            language = "python"
            name = "python3"
        })
    }

    $notebookJson = $notebook | ConvertTo-Json -Depth 100
    Write-Utf8NoBom -Path $Path -Content (ConvertTo-AsciiJson -Content $notebookJson)
}

function ConvertTo-KaggleSlug {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $slug = $Value.ToLowerInvariant()
    $slug = [System.Text.RegularExpressions.Regex]::Replace($slug, "[^a-z0-9]+", "-")
    $slug = $slug.Trim("-")
    return $slug
}

function Assert-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $CommandName"
    }
}

function Resolve-KaggleCommand {
    # Prefer the project virtual environment so the sync command works from a
    # regular PowerShell session without requiring `env\Scripts\Activate.ps1`.
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $virtualEnvironmentCandidates = @(
        (Join-Path -Path $projectRoot -ChildPath "env\Scripts\kaggle.exe"),
        (Join-Path -Path $projectRoot -ChildPath ".venv\Scripts\kaggle.exe")
    )
    foreach ($candidate in $virtualEnvironmentCandidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $command = Get-Command "kaggle" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Path
    }

    throw (
        "Kaggle CLI not found. Install it in the project environment with " +
        ".\\env\\Scripts\\python.exe -m pip install kaggle, or add a kaggle executable to PATH."
    )
}

function Assert-File {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

$kaggleCommand = $null
if (-not $SkipPush) {
    $kaggleCommand = Resolve-KaggleCommand
}
Assert-File -Path $NotebookPath -Label "Notebook"
Assert-File -Path $MetadataPath -Label "Kaggle metadata"

if (-not (Test-Path -LiteralPath $BundlePath -PathType Container)) {
    New-Item -ItemType Directory -Path $BundlePath -Force | Out-Null
}

# Load metadata once so we can validate placeholders and keep the code file aligned with the copied notebook.
$metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json

$requiredFields = @("id", "title", "language", "kernel_type")
foreach ($field in $requiredFields) {
    if ([string]::IsNullOrWhiteSpace($metadata.$field)) {
        throw "Missing required metadata field: $field"
    }
}

if ($metadata.id -match "^YOUR_KAGGLE_USERNAME/") {
    throw "Update kaggle/modelisation/kernel-metadata.json with your Kaggle username and kernel slug before pushing."
}

if ($metadata.dataset_sources -contains "YOUR_KAGGLE_USERNAME/YOUR_DATASET_SLUG") {
    throw "Update kaggle/modelisation/kernel-metadata.json with your Kaggle dataset slug before pushing."
}

$kernelSlug = $metadata.id.Split("/", 2)[1]
$titleSlug = ConvertTo-KaggleSlug -Value $metadata.title
if ($kernelSlug -ne $titleSlug) {
    throw "Kaggle metadata mismatch: title resolves to '$titleSlug' but id uses '$kernelSlug'. Align the title or id before pushing."
}

$codeFileName = Split-Path -Path $NotebookPath -Leaf
$targetNotebookPath = Join-Path -Path $BundlePath -ChildPath $codeFileName

# Copy the local notebook into the Kaggle bundle that will be pushed remotely.
Copy-Item -LiteralPath $NotebookPath -Destination $targetNotebookPath -Force

# Kaggle execution expects a kernelspec entry in the notebook metadata.
Ensure-NotebookKernelspec -Path $targetNotebookPath

# Keep the metadata code_file in sync with the copied notebook filename.
$metadata.code_file = $codeFileName
$metadataJson = $metadata | ConvertTo-Json -Depth 20
Write-Utf8NoBom -Path $MetadataPath -Content (ConvertTo-AsciiJson -Content $metadataJson)

Write-Host "Notebook copied to $targetNotebookPath"

if ($SkipPush) {
    Write-Host "SkipPush enabled: bundle prepared but not uploaded."
    exit 0
}

$pushArgs = @("kernels", "push", "-p", $BundlePath)
if (-not [string]::IsNullOrWhiteSpace($Accelerator)) {
    $pushArgs += @("--accelerator", $Accelerator)
}

Write-Host "Running: $kaggleCommand $($pushArgs -join ' ')"
& $kaggleCommand @pushArgs

if ($LASTEXITCODE -ne 0) {
    throw "Kaggle push failed."
}
