#pragma once
#include <string>

// windows dll export/import
#ifdef _WIN32
  #ifdef COLMAP_API_EXPORTS
    #define COLMAP_API __declspec(dllexport)
  #else
    #define COLMAP_API __declspec(dllimport)
  #endif
#else
  #define COLMAP_API
#endif

namespace colmap_api {

/**
 * [功能描述]：运行自动重建。
 * @param _workspace_path：[参数说明] 工作空间路径。
 * @return [返回值说明]：返回 0 表示成功，非 0 表示失败（若失败，out_error 会包含简要说明）。
 */
COLMAP_API int RunAutomaticReconstructor(std::string _workspace_path);

} // namespace colmap_api
