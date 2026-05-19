# compile.ps1 — Compile fire_risk_report.tex to PDF
# Run from the report/ directory or adjust $reportDir below.

$reportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$tex       = Join-Path $reportDir "fire_risk_report.tex"
$name      = "fire_risk_report"

# Locate pdflatex / xelatex
$compilers = @(
    "pdflatex",
    "xelatex",
    "C:\Users\user\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
    "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
)
$compiler = $null
foreach ($c in $compilers) {
    if (Get-Command $c -ErrorAction SilentlyContinue) { $compiler = $c; break }
    if (Test-Path $c) { $compiler = $c; break }
}

if (-not $compiler) {
    Write-Error "No LaTeX compiler found. Install MiKTeX or TeX Live."
    exit 1
}

Write-Host "Using compiler: $compiler" -ForegroundColor Cyan
Set-Location $reportDir

# Pass 1
Write-Host "`n[1/2] First pass..." -ForegroundColor Yellow
& $compiler -interaction=nonstopmode -shell-escape $tex

# Pass 2 (resolves cross-references)
Write-Host "`n[2/2] Second pass..." -ForegroundColor Yellow
& $compiler -interaction=nonstopmode -shell-escape $tex

$pdf = Join-Path $reportDir "$name.pdf"
if (Test-Path $pdf) {
    Write-Host "`nSuccess! PDF created: $pdf" -ForegroundColor Green
    Start-Process $pdf
} else {
    Write-Host "`nPDF not found — check $name.log for errors." -ForegroundColor Red
}
