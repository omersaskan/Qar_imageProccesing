# Meshysiz Environment Setup & Diagnostic (V2)
Write-Output "--- Setting up Environment ---"

# 1. Add COLMAP to PATH
$colmapDir = "C:\colmap\bin"
if (Test-Path $colmapDir) {
    if ($env:PATH -notlike "*$colmapDir*") {
        $env:PATH = "$colmapDir;" + $env:PATH
        Write-Output "[OK] Added COLMAP to PATH"
    }
} else {
    Write-Output "[WARN] COLMAP directory not found at $colmapDir"
}

# 2. Add Blender to PATH
$blenderExe = Get-ChildItem -Path "C:\Program Files\Blender Foundation" -Filter "blender.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if ($blenderExe) {
    $blenderDir = Split-Path $blenderExe
    if ($env:PATH -notlike "*$blenderDir*") {
        $env:PATH = "$blenderDir;" + $env:PATH
        Write-Output "[OK] Added Blender to PATH ($blenderDir)"
    }
} else {
    Write-Output "[WARN] Blender executable not found in C:\Program Files\Blender Foundation"
}

# 3. Add Node.js to PATH
$nodeDir = "C:\Program Files\nodejs"
if (Test-Path $nodeDir) {
    if ($env:PATH -notlike "*$nodeDir*") {
        $env:PATH = "$nodeDir;" + $env:PATH
        Write-Output "[OK] Added Node.js to PATH"
    }
} else {
    Write-Output "[WARN] Node.js directory not found at $nodeDir"
}

# 3.1 Add Global NPM Bin to PATH
$npmGlobalDir = "$env:APPDATA\npm"
if (Test-Path $npmGlobalDir) {
    if ($env:PATH -notlike "*$npmGlobalDir*") {
        $env:PATH = "$npmGlobalDir;" + $env:PATH
        Write-Output "[OK] Added Global NPM Bin to PATH"
    }
}

# 4. Setup Python Alias
function python { py @args }
Write-Output "[OK] Aliased 'python' to 'py'"

# 5. Diagnostic Check
Write-Output "`n--- Diagnostic ---"
py --version
if (Get-Command colmap -ErrorAction SilentlyContinue) { 
    Write-Output "[OK] colmap found: $(colmap -h | Select-Object -First 1)" 
}
if (Get-Command blender -ErrorAction SilentlyContinue) { 
    Write-Output "[OK] blender found: $(blender --version | Select-Object -First 1)" 
}
if (Get-Command npm -ErrorAction SilentlyContinue) { 
    Write-Output "[OK] npm found: $(npm --version)" 
    if (-not (Get-Command gltf-transform -ErrorAction SilentlyContinue)) {
        Write-Output "[HINT] gltf-transform is missing. Run: npm install -g @gltf-transform/cli"
    } else {
        Write-Output "[OK] gltf-transform found"
    }
}
