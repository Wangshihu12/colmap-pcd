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
  return AutomaticReconstructor(_workspace_path);
}

} // namespace colmap_api
