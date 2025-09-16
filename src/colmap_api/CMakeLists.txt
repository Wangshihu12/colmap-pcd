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

# 链接依赖
target_link_libraries(colmap_api
    PRIVATE
        colmap
)

# Windows特定设置
if(WIN32)
    target_compile_definitions(colmap_api
        PRIVATE
            COLMAP_API_BUILD
            NOMINMAX
    )
    
    # 安装规则
    install(TARGETS colmap_api
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
else()
    install(TARGETS colmap_api
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

# 安装头文件
install(FILES colmap_api.h
    DESTINATION include/colmap_api
)