# src/api/CMakeLists.txt
# cmake_minimum_required(VERSION 3.2)

# 创建API库
add_library(colmap_api SHARED
    colmap_api.cpp
)

# 设置目标属性
set_target_properties(colmap_api PROPERTIES
    OUTPUT_NAME "colmap_api"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# 包含目录
target_include_directories(colmap_api
    PUBLIC 
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Windows特定设置
if(WIN32)
    target_compile_definitions(colmap_api
        PRIVATE
            COLMAP_API_BUILD
            COLMAP_API_EXPORTS  # 用于导出符号
            NOMINMAX
    )
    
    # 设置链接选项 - 针对混合链接进行优化
    set_target_properties(colmap_api PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
        # 启用增量链接以加快构建速度
        LINK_FLAGS "/INCREMENTAL:NO /LTCG"
    )
endif()

# 链接依赖 - 分别处理静态和动态库
target_link_libraries(colmap_api
    PRIVATE
        # 主要的colmap库（静态）
        colmap
        
        # COLMAP内部静态库
        ${COLMAP_INTERNAL_LIBRARIES}
)

# 静态链接的外部库 - 使用WHOLEARCHIVE确保所有符号都包含
if(WIN32)
    # 对于每个静态库，使用WHOLEARCHIVE
    target_link_libraries(colmap_api PRIVATE ${COLMAP_STATIC_LIBRARIES})
else()
    # Linux下使用 --whole-archive
    target_link_libraries(colmap_api PRIVATE 
        -Wl,--whole-archive
        ${COLMAP_STATIC_LIBRARIES}
        -Wl,--no-whole-archive
    )
endif()

# 动态链接的外部库
target_link_libraries(colmap_api
    PRIVATE
        ${COLMAP_DYNAMIC_LIBRARIES}
)

# 安装规则
if(WIN32)
    install(TARGETS colmap_api
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    
    # 复制必要的动态库到安装目录
    if(FREEIMAGE_LIBRARIES)
        get_filename_component(FREEIMAGE_DLL_DIR "${FREEIMAGE_LIBRARIES}" DIRECTORY)
        file(GLOB FREEIMAGE_DLLS "${FREEIMAGE_DLL_DIR}/../bin/*.dll")
        install(FILES ${FREEIMAGE_DLLS} 
            DESTINATION bin 
            OPTIONAL
        )
    endif()
else()
    install(TARGETS colmap_api
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

# 安装头文件
install(FILES colmap_api.h export.h
    DESTINATION include/colmap_api
)