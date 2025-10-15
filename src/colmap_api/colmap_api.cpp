// 实现文件包含 COLMAP 内部头
#include "colmap_api.h"
#include "controllers/automatic_reconstruction.h"
#include "base/reconstruction.h"
#include "exe/sfm.h"            // （可选，用于参考）
#include "util/opengl_utils.h"  // kUseOpenGL, RunThreadWithOpenGLContext
#include <iostream>

using namespace colmap;

namespace colmap_api {

/**
 * [功能描述]：实现将 AutomaticReconstructionOptions 映射到 COLMAP 并执行
 * @return [返回值说明]：0 成功，非 0 错误。
 */
int RunAutomaticReconstructor(std::string _workspace_path) {
  return AutomaticReconstructor(_workspace_path, nullptr);
}

/**
 * [功能描述]：运行自动重建（带进度回调）
 * @param _workspace_path：工作空间路径
 * @param callback：进度回调函数
 * @return 0 成功，非 0 错误，-1 用户取消
 */
 int RunAutomaticReconstructorWithCallback(std::string _workspace_path, ReconstructionProgressCallback callback) {
  return AutomaticReconstructor(_workspace_path, callback);
}

double GetReconstructionProgress() {
  return GetReconstructionProgress();
}

bool IsReconstructionRunning() {
  return IsReconstructionRunning();
}

void ResetReconstructionProgress() {
  ResetReconstructionProgress();
}

} // namespace colmap_api
