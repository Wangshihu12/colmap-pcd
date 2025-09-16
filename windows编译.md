cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake `
           -DPCL_DIR=C:/src/vcpkg/installed/x64-windows/share/pcl `
           -DOpenCV_DIR=C:/src/vcpkg/installed/x64-windows/share/opencv `
           -DEigen3_DIR=C:/src/vcpkg/installed/x64-windows/share/eigen3 `
           -DCMAKE_PREFIX_PATH=C:/src/vcpkg/installed/x64-windows `
           -Dgflags_DIR=C:/src/vcpkg/installed/x64-windows/share/gflags `
           -Dglog_DIR=C:/src/vcpkg/installed/x64-windows/share/glog `
           -DCMAKE_INSTALL_PREFIX=F:\github\colmap-pcd\install

cmake --build . --config release --parallel 24

cmake --install . --config Release