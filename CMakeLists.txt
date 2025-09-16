# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.1)

# 定义项目名称和支持的编程语言（C和C++）
project(COLMAP LANGUAGES C CXX)

# 设置COLMAP版本信息
set(COLMAP_VERSION "3.8")
set(COLMAP_VERSION_NUMBER "3800")

# 设置CUDA架构支持所有主要架构
set(CMAKE_CUDA_ARCHITECTURES "all-major")

################################################################################
# 包含CMake依赖项
################################################################################

# 设置CMake模块路径为当前源码目录下的cmake文件夹
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# 包含检查C++编译器标志的模块
include(CheckCXXCompilerFlag)

# 包含辅助宏和命令，并允许包含的文件覆盖此文件中的CMake策略
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)


################################################################################
# 编译选项配置
################################################################################
# SIMD优化选项：启用单指令多数据流优化
option(SIMD_ENABLED "Whether to enable SIMD optimizations" ON)
# OpenMP并行化选项：启用OpenMP多线程并行计算
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)
# 过程间优化选项：启用链接时优化
option(IPO_ENABLED "Whether to enable interprocedural optimization" ON)
# CUDA支持选项：如果可用则启用CUDA加速
option(CUDA_ENABLED "Whether to enable CUDA, if available" OFF)
# 图形用户界面选项：启用图形界面
option(GUI_ENABLED "Whether to enable the graphical UI" ON)
# OpenGL支持选项：如果可用则启用OpenGL渲染
option(OPENGL_ENABLED "Whether to enable OpenGL, if available" ON)
# 测试构建选项：构建测试二进制文件
option(TESTS_ENABLED "Whether to build test binaries" OFF)
# 地址消毒器选项：启用内存错误检测
option(ASAN_ENABLED "Whether to enable AddressSanitizer flags" OFF)
# 性能分析选项：启用Google性能分析工具
option(PROFILING_ENABLED "Whether to enable google-perftools linker flags" OFF)
# 编译器缓存选项：如果可用则启用编译缓存加速
option(CCACHE_ENABLED "Whether to enable compiler caching, if available" ON)
# CGAL库选项：启用计算几何算法库
option(CGAL_ENABLED "Whether to enable the CGAL library" ON)
# Boost静态库选项：使用静态链接的Boost库
option(BOOST_STATIC "Whether to enable static boost library linker flags" ON)
# 激光雷达测试选项：启用测试图像实时显示
option(TEST_LIDAR_ENABLED "Whether to enable test images real-time display" ON)
# API支持选项：启用COLMAP API
option(BUILD_API "Build COLMAP API library" ON)

# 如果启用了测试，则启用CMake测试功能
if(TESTS_ENABLED)
    enable_testing()
endif()
# Boost库链接配置
if(BOOST_STATIC)
    # 如果启用静态链接，设置Boost使用静态库
    set(Boost_USE_STATIC_LIBS ON)
else()
    # 否则添加动态链接定义
    add_definitions("-DBOOST_TEST_DYN_LINK")
endif()

# 构建位置无关代码，以便共享库可以链接到COLMAP的静态库
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# 查找依赖包
################################################################################
# 如果启用了OpenMP，查找OpenMP包
if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()

# 查找Ceres求解器库（非线性优化库）
find_package(Ceres REQUIRED)

# 查找Boost库，包含所需的组件
find_package(Boost REQUIRED COMPONENTS
             program_options  # 程序选项解析
             filesystem       # 文件系统操作
             graph           # 图算法
             system          # 系统相关功能
             unit_test_framework)  # 单元测试框架

# 查找Eigen3线性代数库
find_package(Eigen3 REQUIRED)
# 设置Qt5路径（macOS Homebrew安装路径）
set(Qt5_DIR /opt/homebrew/Cellar/qt@5/5.15.5_1/lib/cmake/Qt5)
# 查找PCL点云库
find_package(PCL REQUIRED)
# 如果启用了激光雷达测试，查找OpenCV库
if (TEST_LIDAR_ENABLED)
    find_package(OpenCV REQUIRED)
    add_definitions(-DTEST_LIDAR_ENABLED)
endif()
# 查找FreeImage图像处理库
find_package(FreeImage REQUIRED)

# 查找FLANN快速最近邻搜索库
find_package(FLANN REQUIRED)
# 查找LZ4压缩库
find_package(LZ4 REQUIRED)

# 查找Metis图分割库
find_package(Metis REQUIRED)

# 查找Google日志库
find_package(Glog REQUIRED)

# 查找SQLite3数据库库
find_package(SQLite3 REQUIRED)

# 添加YAML-CPP支持（YAML配置文件解析库）
find_package(yaml-cpp REQUIRED)

# 设置OpenGL库偏好为GLVND（现代Linux OpenGL库）
set(OpenGL_GL_PREFERENCE GLVND)
# 查找OpenGL库
find_package(OpenGL REQUIRED)
# 查找GLEW OpenGL扩展库
find_package(Glew REQUIRED)
# 查找Git版本控制工具
find_package(Git)
# CGAL库配置
if(CGAL_ENABLED)
    # 禁用CGAL关于CMake构建类型的警告
    set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
    # 我们不使用CGAL数据。这可以防止CMake产生不必要的警告
    set(CGAL_DATA_DIR "unused")
    find_package(CGAL QUIET)
endif()

# 设置CUDA最低版本要求
set(CUDA_MIN_VERSION "7.0")
# CUDA支持配置
if(CUDA_ENABLED)
    # 根据CMake版本选择不同的CUDA查找方式
    if(CMAKE_VERSION VERSION_LESS 3.17)
        # 对于较旧的CMake版本，使用传统的CUDA模块
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "Found CUDA version ${CUDA_VERSION} installed in "
                    "${CUDA_TOOLKIT_ROOT_DIR} via legacy CMake (<3.17) module. "
                    "Using the legacy CMake module means that any installation of "
                    "COLMAP will require that the CUDA libraries are "
                    "available under LD_LIBRARY_PATH.")
            enable_language(CUDA)

            # 定义宏来创建导入的CUDA目标
            macro(declare_imported_cuda_target module)
                add_library(CUDA::${module} INTERFACE IMPORTED)
                target_include_directories(
                    CUDA::${module} INTERFACE ${CUDA_INCLUDE_DIRS})
                target_link_libraries(
                    CUDA::${module} INTERFACE ${CUDA_${module}_LIBRARY} ${ARGN})
            endmacro()

            # 声明CUDA运行时和随机数生成库目标
            declare_imported_cuda_target(cudart ${CUDA_LIBRARIES})
            declare_imported_cuda_target(curand ${CUDA_LIBRARIES})
            
            # 设置CUDA工具包版本和二进制目录
            set(CUDAToolkit_VERSION "${CUDA_VERSION_STRING}")
            set(CUDAToolkit_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
        endif()
    else()
        # 对于较新的CMake版本，使用CUDAToolkit模块
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_FOUND ON)
            enable_language(CUDA)
        endif()
    endif()
endif()
# GUI和Qt5配置
if(GUI_ENABLED)
    # 查找Qt5库，需要Core、OpenGL和Widgets组件，最低版本5.4
    find_package(Qt5 5.4 COMPONENTS Core OpenGL Widgets)
    if(Qt5_FOUND)
        message(STATUS "Found Qt")
        message(STATUS "  Module : ${Qt5Core_DIR}")
        message(STATUS "  Module : ${Qt5OpenGL_DIR}")
        message(STATUS "  Module : ${Qt5Widgets_DIR}")
        # 禁用Qt关键字以避免与C++关键字冲突
        add_definitions(-DQT_NO_KEYWORDS)
    else()
        # 如果找不到Qt5，则禁用GUI
        set(GUI_ENABLED OFF)
    endif()
endif()

# 如果启用了OpenGL但没有启用GUI，则禁用OpenGL
if(OPENGL_ENABLED AND NOT GUI_ENABLED)
    message(STATUS "Disabling GUI will also disable OpenGL")
    set(OPENGL_ENABLED OFF)
endif()

# CGAL库信息输出
if(CGAL_FOUND)
    list(APPEND CGAL_LIBRARY ${CGAL_LIBRARIES})
    message(STATUS "Found CGAL")
    message(STATUS "  Includes : ${CGAL_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${CGAL_LIBRARY}")
endif()


################################################################################
# 编译器特定配置
################################################################################

# 构建类型配置
if(CMAKE_BUILD_TYPE)
    message(STATUS "Build type specified as ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type not specified, using Release")
    set(CMAKE_BUILD_TYPE Release)
    set(IS_DEBUG OFF)
endif()

# Microsoft Visual C++编译器配置
if(IS_MSVC)
    # 一些Glog库的修复
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    # 启用C++异常处理
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # 在Visual Studio中启用对象级并行构建
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

# GNU GCC编译器配置
if(IS_GNU)
    # 检查GCC版本，4.8或更早版本不支持
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "GCC version 4.8 or older not supported")
    endif()

    # 隐藏GCC下Eigen变量未初始化的错误警告
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

# 调试模式配置
if(IS_DEBUG)
    # 在调试模式下，用NaN初始化Eigen矩阵以便检测未初始化使用
    add_definitions("-DEIGEN_INITIALIZE_MATRICES_BY_NAN")
endif()

# SIMD优化支持配置
if(SIMD_ENABLED)
    message(STATUS "Enabling SIMD support")
else()
    message(STATUS "Disabling SIMD support")
endif()

# OpenMP并行化支持配置
if(OPENMP_ENABLED AND OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Disabling OpenMP support")
endif()

# 过程间优化（IPO）配置
if(IPO_ENABLED AND NOT IS_DEBUG AND NOT IS_GNU)
    message(STATUS "Enabling interprocedural optimization")
    set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION 1)
else()
    message(STATUS "Disabling interprocedural optimization")
endif()

# 地址消毒器（AddressSanitizer）配置
if(ASAN_ENABLED)
    message(STATUS "Enabling ASan support")
    if(IS_CLANG OR IS_GNU)
        # 添加地址消毒器编译和链接选项
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope)
        add_link_options(-fsanitize=address)
    else()
        message(FATAL_ERROR "Unsupported compiler for ASan mode")
    endif()
endif()

# CUDA详细配置
if(CUDA_ENABLED AND CUDA_FOUND)
    # 检查CUDA架构是否已定义
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(
            FATAL_ERROR "You must set CMAKE_CUDA_ARCHITECTURES to e.g. 'native', 'all-major', '70', etc. "
            "More information at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html")
    endif()

    # 添加CUDA启用定义
    add_definitions("-DCUDA_ENABLED")

    # 修复某些CUDA和GCC组合的问题（例如在Ubuntu 16.04下）
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
    # 如果架构已弃用，不显示警告
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    # 如果CUDA库函数已弃用，不显示警告
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
    # 为CUDA目标显式设置PIC标志
    if(NOT IS_MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    endif()

    message(STATUS "Enabling CUDA support (version: ${CUDAToolkit_VERSION}, "
                    "archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(CUDA_ENABLED OFF)
    message(STATUS "Disabling CUDA support")
endif()

# GUI支持配置
if(GUI_ENABLED AND Qt5_FOUND)
    add_definitions("-DGUI_ENABLED")
    message(STATUS "Enabling GUI support")
else()
    message(STATUS "Disabling GUI support")
endif()

# OpenGL支持配置
if(OPENGL_ENABLED)
    add_definitions("-DOPENGL_ENABLED")
    message(STATUS "Enabling OpenGL support")
else()
    message(STATUS "Disabling OpenGL support")
endif()

# 编译器缓存（ccache）配置
if(CCACHE_ENABLED)
    find_program(CCACHE ccache)
    if(CCACHE)
        message(STATUS "Enabling ccache support")
        # 设置C和C++编译器启动器为ccache以加速编译
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    else()
        message(STATUS "Disabling ccache support")
    endif()
else()
    message(STATUS "Disabling ccache support")
endif()

# 性能分析支持配置
if(PROFILING_ENABLED)
    message(STATUS "Enabling profiling support")
    # 添加Google性能分析器和内存分配器链接选项
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lprofiler -ltcmalloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler -ltcmalloc")
else()
    message(STATUS "Disabling profiling support")
endif()

# CGAL库支持配置
if(CGAL_FOUND AND CGAL_ENABLED)
    message(STATUS "Enabling CGAL support")
    add_definitions("-DCGAL_ENABLED")
else()
    message(STATUS "Disabling CGAL support")
    set(CGAL_ENABLED OFF)
endif()

# Qt5特殊配置
if(Qt5_FOUND)
    # Qt5是用-reduce-relocations构建的
    if(Qt5_POSITION_INDEPENDENT_CODE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        # Ubuntu 20.04下Qt5 CMake配置错误的解决方案：https://gitlab.kitware.com/cmake/cmake/-/issues/16915
        if(${Qt5_VERSION} VERSION_EQUAL "5.12.8" AND TARGET Qt5::Core)
            get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
            string(REPLACE "-fPIC" "" new_qt5_core_options ${core_options})
            set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_qt5_core_options})
            set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
            if(NOT IS_MSVC)
                set(CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC")
            endif()
        endif()
    endif()

    # 启用Qt资源文件的自动编译
    set(CMAKE_AUTORCC ON)
endif()

################################################################################
# 添加源码
################################################################################

# 生成包含版本定义的源文件
include(GenerateVersionDefinitions)

# 设置COLMAP包含目录列表
set(COLMAP_INCLUDE_DIRS
    ${Boost_INCLUDE_DIRS}        # Boost库头文件目录
    ${EIGEN3_INCLUDE_DIRS}       # Eigen3线性代数库头文件目录
    ${PCL_INCLUDE_DIRS}          # PCL点云库头文件目录
    ${GLOG_INCLUDE_DIRS}         # Google日志库头文件目录
    ${FLANN_INCLUDE_DIRS}        # FLANN最近邻搜索库头文件目录
    ${LZ4_INCLUDE_DIRS}          # LZ4压缩库头文件目录
    ${FREEIMAGE_INCLUDE_DIRS}    # FreeImage图像处理库头文件目录
    ${CERES_INCLUDE_DIRS}        # Ceres求解器库头文件目录
    ${METIS_INCLUDE_DIRS}        # Metis图分割库头文件目录
    ${GLEW_INCLUDE_DIRS}         # GLEW OpenGL扩展库头文件目录
    ${SQLite3_INCLUDE_DIRS}      # SQLite3数据库库头文件目录
)

# 设置COLMAP链接目录列表
set(COLMAP_LINK_DIRS
    ${Boost_LIBRARY_DIRS}        # Boost库目录
)

# 设置COLMAP外部库列表
set(COLMAP_EXTERNAL_LIBRARIES
    ${CMAKE_DL_LIBS}             # 动态链接库
    ${Boost_FILESYSTEM_LIBRARY}  # Boost文件系统库
    ${Boost_PROGRAM_OPTIONS_LIBRARY}  # Boost程序选项库
    ${Boost_SYSTEM_LIBRARY}      # Boost系统库
    ${GLOG_LIBRARIES}            # Google日志库
    ${PCL_LIBRARIES}             # PCL点云库
    ${FLANN_LIBRARIES}           # FLANN最近邻搜索库
    ${LZ4_LIBRARIES}             # LZ4压缩库
    ${FREEIMAGE_LIBRARIES}       # FreeImage图像处理库
    ${METIS_LIBRARIES}           # Metis图分割库
    ${CERES_LIBRARIES}           # Ceres求解器库
    ${OPENGL_LIBRARIES}          # OpenGL库
    ${SQLite3_LIBRARIES}         # SQLite3数据库库
    yaml-cpp                     # YAML-CPP配置文件解析库
)

# 如果找到OpenMP，添加到外部库列表
if(OPENMP_FOUND)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif()

# 如果找到Qt5，添加Qt5相关库和头文件目录
if(Qt5_FOUND)
    list(APPEND COLMAP_INCLUDE_DIRS ${Qt5Core_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${Qt5Core_LIBRARIES} ${Qt5OpenGL_LIBRARIES} ${Qt5Widgets_LIBRARIES})
endif()

# 如果找到CGAL，添加CGAL相关库和头文件目录
if(CGAL_FOUND)
    list(APPEND COLMAP_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS} ${GMP_INCLUDE_DIR})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${CGAL_LIBRARY} ${GMP_LIBRARIES})
    list(APPEND COLMAP_LINK_DIRS ${CGAL_LIBRARIES_DIR})
endif()

# 如果找到OpenCV，添加OpenCV相关库和头文件目录
if(OpenCV_FOUND)
    list(APPEND COLMAP_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${OpenCV_LIBS})
endif()
# 在Unix系统上添加pthread库
if(UNIX)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES pthread)
endif()
# 设置COLMAP内部库列表
set(COLMAP_INTERNAL_LIBRARIES
    lsd              # LSD线段检测库
    pba              # PBA并行束调整库
    poisson_recon    # Poisson重建库
    sift_gpu         # SIFT GPU加速库
    vlfeat           # VLFeat计算机视觉库
)

# 设置包含目录
include_directories(
    lib                    # 库文件目录
    src                    # 源码目录
    ${COLMAP_INCLUDE_DIRS} # 外部库头文件目录
)

# 设置链接目录
link_directories(${COLMAP_LINK_DIRS})

# 添加子目录到构建系统
add_subdirectory(lib)  # 添加库子目录
add_subdirectory(src)  # 添加源码子目录


################################################################################
# 为Visual Studio、XCode等IDE生成源码组
################################################################################

# 添加库源码组
COLMAP_ADD_SOURCE_DIR(lib/LSD LIB_LSD_SRCS *.h *.c)                    # LSD线段检测库源码
COLMAP_ADD_SOURCE_DIR(lib/PBA LIB_PBA_SRCS *.h *.cpp *.cu)             # PBA并行束调整库源码
COLMAP_ADD_SOURCE_DIR(lib/PoissonRecon LIB_POISSON_RECON_SRCS *.h *.cpp *.inl)  # Poisson重建库源码
COLMAP_ADD_SOURCE_DIR(lib/SiftGPU LIB_SIFT_GPU_SRCS *.h *.cpp *.cu)    # SIFT GPU库源码
COLMAP_ADD_SOURCE_DIR(lib/VLFeat LIB_VLFEAT_SRCS *.h *.c *.tc)         # VLFeat库源码

# 添加主要源码组
COLMAP_ADD_SOURCE_DIR(src/base BASE_SRCS *.h *.cc)                     # 基础库源码
COLMAP_ADD_SOURCE_DIR(src/controllers CONTROLLERS_SRCS *.h *.cc)       # 控制器源码
COLMAP_ADD_SOURCE_DIR(src/estimators ESTIMATORS_SRCS *.h *.cc)         # 估计器源码
COLMAP_ADD_SOURCE_DIR(src/exe EXE_SRCS *.h *.cc)                       # 可执行文件源码
COLMAP_ADD_SOURCE_DIR(src/feature FEATURE_SRCS *.h *.cc)               # 特征提取源码
COLMAP_ADD_SOURCE_DIR(src/mvs MVS_SRCS *.h *.cc *.cu)                  # 多视图立体视觉源码
COLMAP_ADD_SOURCE_DIR(src/optim OPTIM_SRCS *.h *.cc)                   # 优化算法源码
COLMAP_ADD_SOURCE_DIR(src/retrieval RETRIEVAL_SRCS *.h *.cc)           # 检索算法源码
COLMAP_ADD_SOURCE_DIR(src/sfm SFM_SRCS *.h *.cc)                       # 结构从运动源码
COLMAP_ADD_SOURCE_DIR(src/tools TOOLS_SRCS *.h *.cc)                   # 工具源码
COLMAP_ADD_SOURCE_DIR(src/ui UI_SRCS *.h *.cc)                         # 用户界面源码
COLMAP_ADD_SOURCE_DIR(src/util UTIL_SRCS *.h *.cc)                     # 工具函数源码
COLMAP_ADD_SOURCE_DIR(src/lidar LIDAR_SRCS *.h *.cc)                   # 激光雷达源码

# 将所有源文件添加到常规库目标，因为使用自定义目标不允许我们设置其C++包含目录
# （因此智能感知无法找到任何包含的文件）
add_library(
    ${COLMAP_SRC_ROOT_FOLDER}    # COLMAP源码根文件夹目标
    ${LIB_LSD_SRCS}              # LSD库源码
    ${LIB_PBA_SRCS}              # PBA库源码
    ${LIB_POISSON_RECON_SRCS}    # Poisson重建库源码
    ${LIB_SIFT_GPU_SRCS}         # SIFT GPU库源码
    ${LIB_VLFEAT_SRCS}           # VLFeat库源码
    ${BASE_SRCS}                 # 基础库源码
    ${CONTROLLERS_SRCS}          # 控制器源码
    ${ESTIMATORS_SRCS}           # 估计器源码
    ${EXE_SRCS}                  # 可执行文件源码
    ${FEATURE_SRCS}              # 特征提取源码
    ${MVS_SRCS}                  # 多视图立体视觉源码
    ${OPTIM_SRCS}                # 优化算法源码
    ${RETRIEVAL_SRCS}            # 检索算法源码
    ${SFM_SRCS}                  # 结构从运动源码
    ${TOOLS_SRCS}                # 工具源码
    ${UI_SRCS}                   # 用户界面源码
    ${UTIL_SRCS}                 # 工具函数源码
    ${LIDAR_SRCS}                # 激光雷达源码
)

# 防止库被自动编译
set_target_properties(
    ${COLMAP_SRC_ROOT_FOLDER} 
    PROPERTIES                
    EXCLUDE_FROM_ALL 1           # 从所有目标中排除
    EXCLUDE_FROM_DEFAULT_BUILD 1)  # 从默认构建中排除


################################################################################
# 安装和卸载脚本
################################################################################

# 安装头文件
install(DIRECTORY src/     
        DESTINATION include/colmap
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY lib/
        DESTINATION include/colmap/lib
        FILES_MATCHING REGEX ".*[.]h|.*[.]hpp|.*[.]inl")

# 生成并安装CMake配置文件
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfig.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfig.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfig.cmake"
        DESTINATION "share/colmap")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfigVersion.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfigVersion.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/COLMAPConfigVersion.cmake"
        DESTINATION "share/colmap")

# 安装依赖项的find_package脚本
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
        DESTINATION share/colmap
        FILES_MATCHING PATTERN "Find*.cmake")

# 在Windows下安装批处理脚本
if(IS_MSVC)
    install(FILES "scripts/shell/COLMAP.bat" "scripts/shell/RUN_TESTS.bat"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION "/")
endif()

# 在Linux/Unix下安装应用程序菜单项
if(UNIX AND NOT APPLE)
    install(FILES "doc/COLMAP.desktop" DESTINATION "share/applications")
endif()

# 配置卸载脚本
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeUninstall.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake"
               IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)
set_target_properties(uninstall PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})
