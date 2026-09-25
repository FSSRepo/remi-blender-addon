#Requires -Version 5.1
<#
.SYNOPSIS
    Builds the Remi Blender extension archive for Windows x64.

.DESCRIPTION
    Packages the repository into dist\remi-<version>-windows-x64.zip and validates
    it with the Blender extension CLI.

.PARAMETER BlenderBin
    Path to blender.exe. Defaults to the Blender 5.1 installation directory.

.PARAMETER OutputDir
    Directory that receives the archive. Defaults to <repo>\dist.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_extension.ps1
#>
[CmdletBinding()]
param(
    [string]$BlenderBin,
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir '..')).Path

if (-not $BlenderBin) {
    $BlenderBin = 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoDir 'dist'
}

if (-not (Test-Path $BlenderBin)) {
    throw "Blender executable not found: $BlenderBin"
}

$manifestPath = Join-Path $RepoDir 'blender_manifest.toml'
$match = Select-String -Path $manifestPath -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $match) {
    throw 'Could not read the extension version from blender_manifest.toml'
}
$Version = $match.Matches[0].Groups[1].Value

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Artifact = Join-Path $OutputDir "remi-$Version-windows-x64.zip"
Remove-Item $Artifact -ErrorAction SilentlyContinue

& $BlenderBin --command extension build --source-dir $RepoDir --output-filepath $Artifact
if ($LASTEXITCODE -ne 0) { throw 'Extension build failed' }

& $BlenderBin --command extension validate $Artifact
if ($LASTEXITCODE -ne 0) { throw 'Extension validation failed' }

Write-Output $Artifact
