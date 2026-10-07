<#
    run_all.ps1
    ---------------------------------------------------------------------------
    Rebuilds the whole Phase 1 pipeline from the raw scrape, in order.

        .\scripts\run_all.ps1            # every stage
        .\scripts\run_all.ps1 -SkipEmbed # everything except the 10-minute embed
        .\scripts\run_all.ps1 -EmbedOnly # just build/repair the embeddings

    Stages:
        1  combine_data   ET_Data_fetch.json/*.json      -> Data/articles_raw.csv
        2  clean_data     raw CSV                       -> Data/articles_clean.csv
        3  check_data     validate the clean dataset
        4  create_embeddings  clean CSV                  -> models/article_embeddings.npy
        5  test_classification  retrieval + grounding checks

    Every step is safe to re-run. create_embeddings resumes from a checkpoint,
    and it is skipped automatically if the matrix already matches the corpus.
#>

[CmdletBinding()]
param(
    [switch]$SkipEmbed,
    [switch]$EmbedOnly,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

function Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ("-" * 72)
    Write-Host "  $Message"
    Write-Host ("-" * 72)
}

function Run {
    param([string[]]$Arguments)
    Write-Host "  > python $($Arguments -join ' ')"
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: python $($Arguments -join ' ')"
    }
}

if (-not (Test-Path $py)) {
    Write-Host "Virtual environment not found. Create it with:"
    Write-Host "    python -m venv .venv"
    Write-Host "    .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

Push-Location $root
try {
    if ($EmbedOnly) {
        Step "Stage 4: embeddings"
        Run @("src\create_embeddings.py", "--resume")
        return
    }

    if (-not $SkipEmbed) {
        Step "Stage 1: combine raw JSON files"
        Run @("src\combine_data.py")

        Step "Stage 2: clean and normalise"
        Run @("src\clean_data.py")

        Step "Stage 3: validate the dataset"
        Run @("src\check_data.py")
    }

    if (-not $SkipEmbed) {
        Step "Stage 4: build embeddings (about 10 minutes on a laptop)"
        Run @("src\create_embeddings.py", "--resume")
    }

    if (-not $SkipTests) {
        Step "Stage 5: retrieval checks"
        Run @("src\test_classification.py")
    }

    Write-Host ""
    Write-Host "Pipeline complete. Launch the app with:"
    Write-Host "    .\start.bat"
}
finally {
    Pop-Location
}