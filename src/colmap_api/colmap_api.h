#pragma once
#include <string>
#include <functional>

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
 * [类型定义]：重建进度回调函数类型（重命名以避免与内部ProgressCallback冲突）
 * @param progress：当前进度百分比（0.0-100.0）
 * @param stage_name：当前阶段名称（如"特征提取"、"特征匹配"等）
 * @param is_finished：是否已完成
 * @return 返回 false 可以取消重建，返回 true 继续执行
 */
using ReconstructionProgressCallback = std::function<bool(double progress, const std::string& stage_name, bool is_finished)>;

/**
 * [功能描述]：运行自动重建。
 * @param _workspace_path：[参数说明] 工作空间路径。
 * @return [返回值说明]：返回 0 表示成功，非 0 表示失败（若失败，out_error 会包含简要说明）。
 */
COLMAP_API int RunAutomaticReconstructor(std::string _workspace_path);

/**
 * [功能描述]：运行自动重建（带进度回调）。
 * @param _workspace_path：工作空间路径。
 * @param callback：进度回调函数，在重建过程中定期调用以报告进度。
 *                  如果回调返回 false，将尝试取消重建。
 * @return 返回 0 表示成功，非 0 表示失败，-1 表示用户取消。
 */
 COLMAP_API int RunAutomaticReconstructorWithCallback(
  std::string _workspace_path, 
  ReconstructionProgressCallback callback);

/**
 * [功能描述]：获取当前重建进度百分比。
 * @return [返回值说明]：进度值（0.0-100.0）。
 *         - 返回 0.0 表示未开始或刚开始
 *         - 返回 100.0 表示已完成
 *         - 中间值表示进行中
 */
 COLMAP_API double GetReconstructionProgress();

 /**
  * [功能描述]：检查重建是否正在运行。
  * @return [返回值说明]：true 表示正在运行，false 表示未运行或已完成。
  */
 COLMAP_API bool IsReconstructionRunning();
 
 /**
  * [功能描述]：重置进度状态（开始新的重建前调用）。
  * @return 无返回值
  */
 COLMAP_API void ResetReconstructionProgress();

} // namespace colmap_api
