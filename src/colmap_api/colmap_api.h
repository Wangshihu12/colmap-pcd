#pragma once
#include "export.h"
#include <string>

namespace colmap_api {

/**
 * [功能描述]：运行自动重建（封装 COLMAP 的 AutomaticReconstructionController）。
 * @param opts：[参数说明] 自动重建参数，见 AutomaticReconstructionOptions。
 * @param out_error：[参数说明] 输出错误信息（可传 nullptr）。
 * @return [返回值说明]：返回 0 表示成功，非 0 表示失败（若失败，out_error 会包含简要说明）。
 */
COLMAP_API int RunAutomaticReconstructor();

} // namespace colmap_api
