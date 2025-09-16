# create_portable.ps1
param(
    [string]$BuildDir = "build",
    [string]$InstallDir = "install"
)

Write-Host "创建COLMAP便携式分发包..."

# 创建目录结构
New-Item -ItemType Directory -Force -Path "$InstallDir\bin"
New-Item -ItemType Directory -Force -Path "$InstallDir\lib" 
New-Item -ItemType Directory -Force -Path "$InstallDir\include"

# 复制主要文件
# Copy-Item "$BuildDir\src\colmap_api\Release\colmap_api.dll" "$InstallDir\bin\"
# Copy-Item "$BuildDir\src\colmap_api\Release\colmap_api.lib" "$InstallDir\lib\"

# 复制头文件
# Copy-Item "src\colmap_api\*.h" "$InstallDir\include\" -Recurse
# Copy-Item "src\" "$InstallDir\include\colmap\" -Recurse -Force

# 查找并复制依赖的DLL
$dependencies = @()

# 使用dumpbin查找依赖
$dumpbin_output = & "dumpbin" /dependents "$InstallDir\bin\colmap_api.dll" 2>$null
foreach($line in $dumpbin_output) {
    if($line -match "\.dll$") {
        $dll = $line.Trim()
        if($dll -notmatch "^(kernel32|user32|msvcr|api-ms-)") {
            $dependencies += $dll
        }
    }
}
$dependencies += "metis.dll"
$dependencies += "gflags.dll"
$dependencies += "boost_iostreams-vc143-mt-x64-1_88.dll"
$dependencies += "pcl_io_ply.dll"
$dependencies += "libpng16.dll"
$dependencies += "zlib1.dll"
$dependencies += "jpeg62.dll"
$dependencies += "tiff.dll"
$dependencies += "openjp2.dll"
$dependencies += "libwebpmux.dll"
$dependencies += "libwebpdecoder.dll"
$dependencies += "raw.dll"
$dependencies += "OpenEXR-3_3.dll"
$dependencies += "libwebp.dll"
$dependencies += "Iex-3_3.dll"
$dependencies += "Imath-3_2.dll"
$dependencies += "libwebpdemux.dll"
$dependencies += "double-conversion.dll"
$dependencies += "pcre2-16.dll"
$dependencies += "bz2.dll"
$dependencies += "liblzma.dll"
$dependencies += "zstd.dll"
$dependencies += "liblzma.dll"
$dependencies += "lcms2-2.dll"
$dependencies += "IlmThread-3_3.dll"
$dependencies += "OpenEXRCore-3_3.dll"
$dependencies += "libsharpyuv.dll"
$dependencies += "harfbuzz.dll"
$dependencies += "deflate.dll"
$dependencies += "freetype.dll"
$dependencies += "brotlidec.dll"
$dependencies += "brotlicommon.dll"

# 输出过滤后的依赖到终端
Write-Host "`n找到的第三方依赖DLL文件:" -ForegroundColor Green
if($dependencies.Count -gt 0) {
    foreach($dep in $dependencies) {
        Write-Host "  - $dep" -ForegroundColor Yellow
    }
    Write-Host "`n总共找到 $($dependencies.Count) 个第三方依赖" -ForegroundColor Cyan
} else {
    Write-Host "  未找到第三方依赖" -ForegroundColor Red
}

# 从系统和vcpkg目录复制依赖DLL
foreach($dll in $dependencies) {
    $found = $false
    
    # 首先在vcpkg目录中查找
    $vcpkg_paths = @("C:\src\vcpkg\installed\x64-windows\bin", "C:\tools\vcpkg\installed\x64-windows\bin")
    foreach($path in $vcpkg_paths) {
        $dll_path = "$path\$dll"
        if(Test-Path $dll_path) {
            Copy-Item $dll_path "$InstallDir\bin\"
            Write-Host "从vcpkg复制依赖: $dll" -ForegroundColor Green
            $found = $true
            break
        }
    }
    
    # 如果在vcpkg中没找到，则在系统PATH中查找
    if(-not $found) {
        $system_paths = $env:PATH -split ';'
        foreach($path in $system_paths) {
            if($path -and (Test-Path $path)) {
                $dll_path = "$path\$dll"
                if(Test-Path $dll_path) {
                    Copy-Item $dll_path "$InstallDir\bin\"
                    Write-Host "从系统路径复制依赖: $dll (来源: $path)" -ForegroundColor Yellow
                    $found = $true
                    break
                }
            }
        }
    }
    
    # 如果都没找到，显示警告
    if(-not $found) {
        Write-Host "警告: 未找到依赖文件 $dll" -ForegroundColor Red
    }
}

# # 验证依赖是否能正确加载
# Write-Host "`n验证依赖加载..." -ForegroundColor Cyan
# $test_dll = "$InstallDir\bin\colmap_api.dll"
# if (Test-Path $test_dll) {
#     try {
#         # 使用dumpbin再次检查依赖，看是否还有缺失
#         $verify_output = & "dumpbin" /dependents $test_dll 2>$null
#         $missing_deps = @()
#         foreach ($line in $verify_output) {
#             if ($line -match "\.dll$") {
#                 $dll = $line.Trim()
#                 if ($dll -notmatch "^(kernel32|user32|msvcr|api-ms-)") {
#                     $dll_path = "$InstallDir\bin\$dll"
#                     if (-not (Test-Path $dll_path)) {
#                         $missing_deps += $dll
#                     }
#                 }
#             }
#         }
        
#         if ($missing_deps.Count -eq 0) {
#             Write-Host "✓ 所有依赖都已正确复制" -ForegroundColor Green
#         } else {
#             Write-Host "⚠ 仍有缺失的依赖:" -ForegroundColor Yellow
#             foreach ($dep in $missing_deps) {
#                 Write-Host "  - $dep" -ForegroundColor Red
#             }
#         }
#     } catch {
#         Write-Host "⚠ 无法验证依赖加载状态" -ForegroundColor Yellow
#     }
# } else {
#     Write-Host "⚠ 未找到colmap_api.dll文件" -ForegroundColor Red
# }

Write-Host "`n便携式分发包创建完成: $InstallDir"