#Requires -Version 5.1
<#
.SYNOPSIS
    Builds the Remi native modules (Instant Meshes + xatlas) for CPython on Windows.

.DESCRIPTION
    Configures and builds both pybind11 extension modules with MSVC and copies the
    resulting .pyd files next to the Python packages that import them.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_native.ps1
#>
[CmdletBinding()]
param(
    [string]$VenvDir,
    [ValidateSet('Release', 'Debug')]
    [string]$Config = 'Release'
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir '..')).Path

if (-not $VenvDir) {
    $VenvDir = Join-Path $RepoDir '.venv'
}
$Python = Join-Path $VenvDir 'python.exe'
$Pybind11Dir = Join-Path $VenvDir 'Lib\site-packages\pybind11\share\cmake\pybind11'

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found: $Python. Create the environment with 'conda create -p .venv python=3.13 -y'."
}
if (-not (Test-Path $Pybind11Dir)) {
    throw "pybind11 CMake package not found: $Pybind11Dir. Install it with '$Python -m pip install pybind11'."
}

$Modules = @(
    @{ Name = 'instant_meshes'; Source = 'instant_meshes\native'; Output = 'instant_meshes\_native' },
    @{ Name = 'uv_mapping';     Source = 'uv_mapping\native';     Output = 'uv_mapping\_native' }
)

foreach ($Module in $Modules) {
    $sourceDir = Join-Path $RepoDir $Module.Source
    $buildDir = Join-Path $sourceDir 'build'
    $outputDir = Join-Path $RepoDir $Module.Output

    Write-Host "Configuring $($Module.Name)"
    $configureArgs = @(
        '-S', $sourceDir,
        '-B', $buildDir,
        '-G', 'Visual Studio 17 2022',
        '-A', 'x64',
        "-DPython_EXECUTABLE=$Python",
        "-Dpybind11_DIR=$Pybind11Dir",
        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'
    )
    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed for $($Module.Name)" }

    Write-Host "Building $($Module.Name) ($Config)"
    & cmake --build $buildDir --config $Config --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed for $($Module.Name)" }

    $artifacts = Get-ChildItem -Path (Join-Path $buildDir $Config) -Filter '*.pyd'
    if (-not $artifacts) {
        throw "No .pyd produced for $($Module.Name) in $(Join-Path $buildDir $Config)"
    }
    foreach ($artifact in $artifacts) {
        Copy-Item $artifact.FullName $outputDir -Force
        Write-Host "Installed $($artifact.Name) -> $($Module.Output)"
    }
}

Write-Host 'Native modules are up to date.'
