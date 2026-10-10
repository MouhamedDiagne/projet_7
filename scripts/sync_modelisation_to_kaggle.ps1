param(
    [string]$NotebookPath = "notebooks/modelisation.ipynb",
    [string]$BundlePath = "kaggle/modelisation",
    [string]$MetadataPath = "kaggle/modelisation/kernel-metadata.json",
    [string]$TrackingPath = "tracking",
    [string]$RequirementsPath = "requirements-kaggle.txt",
    [string]$TrackingRequirementsPath = "requirements-tracking.txt",
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
    Write-Utf8NoBom -Path $Path -Content $notebookJson
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

function Assert-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label not found: $Path"
    }
}

if (-not $SkipPush) {
    Assert-Command -CommandName "kaggle"
}
Assert-File -Path $NotebookPath -Label "Notebook"
Assert-File -Path $MetadataPath -Label "Kaggle metadata"
Assert-Directory -Path $TrackingPath -Label "Tracking package"
Assert-File -Path $RequirementsPath -Label "Kaggle requirements file"
Assert-File -Path $TrackingRequirementsPath -Label "Tracking requirements file"

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

# Ship the shared tracking package and its dependencies with the notebook so
# `/kaggle/working` can import them at runtime. The bundle is generated, so its
# tracking copy is replaced rather than retaining stale source or bytecode files.
$resolvedBundlePath = (Resolve-Path -LiteralPath $BundlePath).Path
$targetTrackingPath = Join-Path -Path $resolvedBundlePath -ChildPath "tracking"
if (Test-Path -LiteralPath $targetTrackingPath) {
    $resolvedTargetTrackingPath = (Resolve-Path -LiteralPath $targetTrackingPath).Path
    $bundlePrefix = $resolvedBundlePath.TrimEnd('\') + '\'
    if (-not $resolvedTargetTrackingPath.StartsWith($bundlePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace a tracking bundle outside the Kaggle bundle: $resolvedTargetTrackingPath"
    }
    Remove-Item -LiteralPath $resolvedTargetTrackingPath -Recurse -Force
}
New-Item -ItemType Directory -Path $targetTrackingPath -Force | Out-Null
Copy-Item -Path (Join-Path -Path $TrackingPath -ChildPath "*") -Destination $targetTrackingPath -Recurse -Force -Exclude "__pycache__", "*.pyc"

$targetRequirementsPath = Join-Path -Path $BundlePath -ChildPath "requirements-kaggle.txt"
Copy-Item -LiteralPath $RequirementsPath -Destination $targetRequirementsPath -Force
$targetTrackingRequirementsPath = Join-Path -Path $BundlePath -ChildPath "requirements-tracking.txt"
Copy-Item -LiteralPath $TrackingRequirementsPath -Destination $targetTrackingRequirementsPath -Force

# Kaggle execution expects a kernelspec entry in the notebook metadata.
Ensure-NotebookKernelspec -Path $targetNotebookPath

# Keep the metadata code_file in sync with the copied notebook filename.
$metadata.code_file = $codeFileName
$metadataJson = $metadata | ConvertTo-Json -Depth 20
Write-Utf8NoBom -Path $MetadataPath -Content $metadataJson

Write-Host "Notebook copied to $targetNotebookPath"

if ($SkipPush) {
    Write-Host "SkipPush enabled: bundle prepared but not uploaded."
    exit 0
}

$pushArgs = @("kernels", "push", "-p", $BundlePath)
if (-not [string]::IsNullOrWhiteSpace($Accelerator)) {
    $pushArgs += @("--accelerator", $Accelerator)
}

Write-Host "Running: kaggle $($pushArgs -join ' ')"
& kaggle @pushArgs

if ($LASTEXITCODE -ne 0) {
    throw "Kaggle push failed."
}
