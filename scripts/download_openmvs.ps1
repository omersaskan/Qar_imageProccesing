$url = "https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip"
$dest = "$env:TEMP\OpenMVS.zip"
Write-Host "Downloading OpenMVS from $url..."
Invoke-WebRequest -Uri $url -OutFile $dest
Write-Host "Download complete. Extracting to C:\OpenMVS..."
if (-not (Test-Path "C:\OpenMVS")) {
    New-Item -ItemType Directory -Force -Path "C:\OpenMVS"
}
Expand-Archive -Path $dest -DestinationPath "C:\OpenMVS" -Force
Remove-Item $dest
Write-Host "Extraction complete. Showing directory contents:"
Get-ChildItem -Path "C:\OpenMVS"
