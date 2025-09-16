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

# 编译器警告级别和C++标准设置
if(IS_MSVC)
    # 如果是Microsoft Visual C++编译器，设置警告级别为/W3（中等警告级别）
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
elseif(IS_GNU OR IS_CLANG)
    # 如果是GNU GCC或Clang编译器，启用所有警告（-Wall）
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    # 我们主要希望使用C++14来支持最新的依赖项，但较旧的GCC版本
    # 与CUDA结合使用时还不支持C++14。在最坏的情况下，这会导致编译错误，
    # 在COLMAP这边我们无能为力。
    if(CUDA_ENABLED AND
       IS_GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6)
        # 如果启用了CUDA且GCC版本小于6，则使用C++11标准
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        # 否则使用C++14标准
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    endif()
endif()

# 添加各个子目录到构建系统中
add_subdirectory(base)        # 基础库模块
add_subdirectory(controllers) # 控制器模块
add_subdirectory(estimators)  # 估计器模块
add_subdirectory(exe)         # 可执行文件模块
add_subdirectory(feature)     # 特征提取模块
add_subdirectory(mvs)         # 多视图立体视觉模块
add_subdirectory(optim)       # 优化模块
add_subdirectory(retrieval)   # 检索模块
add_subdirectory(sfm)         # 结构从运动模块
add_subdirectory(tools)       # 工具模块
add_subdirectory(util)        # 工具函数模块
add_subdirectory(lidar)       # 激光雷达模块
if (GUI_ENABLED)
    # 如果启用了GUI，则添加用户界面模块
    add_subdirectory(ui)
endif()

# 创建主要的COLMAP库
COLMAP_ADD_LIBRARY(colmap ${COLMAP_SOURCES})

# 在定义colmap库的地方，确保符号可见性
if(BUILD_SHARED_LIBS OR BUILD_API)
    # 确保内部符号对API可见
    target_compile_definitions(colmap PRIVATE COLMAP_BUILD_SHARED_LIBS)
endif()

if(BUILD_API)
    add_subdirectory(colmap_api)
endif()

# CUDA相关配置
if(CUDA_ENABLED)
    # 启用CUDA快速数学运算（使用更快的数学函数，但精度可能略低）
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

    # 使用每个线程单独的流，以允许同一设备上多个线程之间的并发内核执行
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

    # 抑制警告：
    # ptxas警告：入口函数X的堆栈大小无法静态确定
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-suppress-stack-size-warning")

    # Ubuntu 16.04的修复
    add_definitions("-D_MWAITXINTRIN_H_INCLUDED")

    if(MSVC)
        # 避免通过<windows.h>引入过多的头文件
        add_definitions("-DWIN32_LEAN_AND_MEAN")
    endif()

    # 创建CUDA库并链接到主库
    COLMAP_ADD_CUDA_LIBRARY(colmap_cuda ${COLMAP_CUDA_SOURCES})
    target_link_libraries(colmap_cuda colmap)

    # 将CUDA库和所有依赖库链接到主库
    target_link_libraries(colmap
        colmap_cuda ${COLMAP_INTERNAL_LIBRARIES} ${COLMAP_EXTERNAL_LIBRARIES})
else()
    # 如果没有启用CUDA，只链接内部和外部库
    target_link_libraries(colmap
        ${COLMAP_INTERNAL_LIBRARIES} ${COLMAP_EXTERNAL_LIBRARIES})
endif()
