cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake `
           -DPCL_DIR=C:/src/vcpkg/installed/x64-windows/share/pcl `
           -DOpenCV_DIR=C:/src/vcpkg/installed/x64-windows/share/opencv `
           -DEigen3_DIR=C:/src/vcpkg/installed/x64-windows/share/eigen3 `
           -DCMAKE_PREFIX_PATH=C:/src/vcpkg/installed/x64-windows `
           -Dgflags_DIR=C:/src/vcpkg/installed/x64-windows/share/gflags `
           -Dglog_DIR=C:/src/vcpkg/installed/x64-windows/share/glog `
           -DCMAKE_INSTALL_PREFIX=F:\github\colmap-pcd\install

cmake --build build --config release --parallel 24

cmake --install . --config Release

# 静态构建
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=F:/vcpkg/scripts/buildsystems/vcpkg.cmake `
           -DVCPKG_TARGET_TRIPLET=x64-windows-static `
           -DPCL_DIR=F:/vcpkg/installed/x64-windows-static/share/pcl `
           -DOpenCV_DIR=F:/vcpkg/installed/x64-windows-static/share/opencv `
           -DEigen3_DIR=F:/vcpkg/installed/x64-windows-static/share/eigen3 `
           -Dgflags_DIR=F:/vcpkg/installed/x64-windows-static/share/gflags `
           -Dglog_DIR=F:/vcpkg/installed/x64-windows-static/share/glog `
           -DCMAKE_INSTALL_PREFIX=F:\github\colmap-pcd\install `
           -DFlann_DIR=F:/vcpkg/installed/x64-windows-static/share/flann

# 混合构建
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=F:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static-md `
  -DCMAKE_PREFIX_PATH="F:/vcpkg/installed/x64-windows-static-md;F:/vcpkg/installed/x64-windows" `
  -DFreeImage_DIR=F:/vcpkg/installed/x64-windows/share/freeimage `
  -DFLANN_DIR=F:/vcpkg/installed/x64-windows-static-md/share/flann `
  -DEigen3_DIR=F:/vcpkg/installed/x64-windows-static-md/share/eigen3 `
  -Dgflags_DIR=F:/vcpkg/installed/x64-windows-static-md/share/gflags `
  -Dglog_DIR=F:/vcpkg/installed/x64-windows-static-md/share/glog `
  -DPCL_DIR=F:/vcpkg/installed/x64-windows-static-md/share/pcl `
  -DOpenCV_DIR=F:/vcpkg/installed/x64-windows-static-md/share/opencv `
  -DCMAKE_INSTALL_PREFIX=F:/github/colmap-pcd/install

cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=F:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static-md `
  -DCMAKE_PREFIX_PATH="F:/vcpkg/installed/x64-windows-static-md;F:/vcpkg/installed/x64-windows" `
  -DFreeImage_DIR=F:/vcpkg/installed/x64-windows/share/freeimage `
  -DFLANN_DIR=F:/vcpkg/installed/x64-windows-static-md/share/flann `
  -DEigen3_DIR=F:/vcpkg/installed/x64-windows-static-md/share/eigen3 `
  -Dgflags_DIR=F:/vcpkg/installed/x64-windows-static-md/share/gflags `
  -Dglog_DIR=F:/vcpkg/installed/x64-windows-static-md/share/glog `
  -DPCL_DIR=F:/vcpkg/installed/x64-windows-static-md/share/pcl `
  -DOpenCV_DIR=F:/vcpkg/installed/x64-windows-static-md/share/opencv `
  -DSQLite3_DIR=F:/vcpkg/installed/x64-windows-static-md/share/sqlite3 `
  -DCMAKE_INSTALL_PREFIX=F:/github/colmap-pcd/install `
  -DCMAKE_BUILD_TYPE=Release