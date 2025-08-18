// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "exe/feature.h"

#include "base/camera_models.h"
#include "base/image_reader.h"
#include "exe/gui.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "util/misc.h"
#include "util/opengl_utils.h"
#include "util/option_manager.h"

namespace colmap {
namespace {

bool VerifyCameraParams(const std::string& camera_model,
                        const std::string& params) {
  if (!ExistsCameraModelWithName(camera_model)) {
    std::cerr << "ERROR: Camera model does not exist" << std::endl;
    return false;
  }

  const std::vector<double> camera_params = CSVToVector<double>(params);
  const int camera_model_id = CameraModelNameToId(camera_model);

  if (camera_params.size() > 0 &&
      !CameraModelVerifyParams(camera_model_id, camera_params)) {
    std::cerr << "ERROR: Invalid camera parameters" << std::endl;
    return false;
  }
  return true;
}

bool VerifySiftGPUParams(const bool use_gpu) {
#if !defined(CUDA_ENABLED) && !defined(OPENGL_ENABLED)
  if (use_gpu) {
    std::cerr << "ERROR: Cannot use Sift GPU without CUDA or OpenGL support; "
                 "set SiftExtraction.use_gpu or SiftMatching.use_gpu to false."
              << std::endl;
    return false;
  }
#endif
  return true;
}

}  // namespace

void UpdateImageReaderOptionsFromCameraMode(ImageReaderOptions& options,
                                            CameraMode mode) {
  switch (mode) {
    case CameraMode::AUTO:
      options.single_camera = false;
      options.single_camera_per_folder = false;
      options.single_camera_per_image = false;
      break;
    case CameraMode::SINGLE:
      options.single_camera = true;
      options.single_camera_per_folder = false;
      options.single_camera_per_image = false;
      break;
    case CameraMode::PER_FOLDER:
      options.single_camera = false;
      options.single_camera_per_folder = true;
      options.single_camera_per_image = false;
      break;
    case CameraMode::PER_IMAGE:
      options.single_camera = false;
      options.single_camera_per_folder = false;
      options.single_camera_per_image = true;
      break;
  }
}
// feature extractor运行函数
/**
 * 运行特征提取器的主函数
 * 
 * 该函数负责从命令行参数解析配置选项，设置图像读取器，
 * 并启动SIFT特征提取过程，支持CPU和GPU两种计算模式
 * 
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return 程序执行状态码
 */
int RunFeatureExtractor(int argc, char** argv) {
  // 声明局部变量
  std::string image_list_path;        // 图像列表文件路径
  int camera_mode = -1;               // 相机模式（-1表示未指定）
  std::string descriptor_normalization = "l1_root";  // 描述子归一化方式，默认为L1_ROOT

  //////////////////////////////////////////////////////////////////////////////
  // 创建和配置选项管理器
  //////////////////////////////////////////////////////////////////////////////
  
  OptionManager options;  // 创建选项管理器对象
  
  // 添加各种配置选项组
  options.AddDatabaseOptions();        // 数据库相关选项（数据库路径等）
  options.AddImageOptions();           // 图像相关选项（图像路径、相机模型等）
  options.AddDefaultOption("camera_mode", &camera_mode);  // 相机模式选项
  options.AddDefaultOption("image_list_path", &image_list_path);  // 图像列表路径选项
  options.AddDefaultOption("descriptor_normalization", &descriptor_normalization,
                           "{'l1_root', 'l2'}");  // 描述子归一化选项，提供两种选择
  options.AddExtractionOptions();      // 特征提取相关选项（SIFT参数等）
  
  // 解析命令行参数，将解析结果存储到相应的变量中
  options.Parse(argc, argv);
  
  //////////////////////////////////////////////////////////////////////////////
  // 配置图像读取器选项
  //////////////////////////////////////////////////////////////////////////////
  
  // 获取图像读取器选项的副本
  ImageReaderOptions reader_options = *options.image_reader;
  
  // 设置数据库路径和图像路径
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  // 如果指定了相机模式，则根据相机模式更新读取器选项
  if (camera_mode >= 0) {
    UpdateImageReaderOptionsFromCameraMode(reader_options,
                                           (CameraMode)camera_mode);
  }

  //////////////////////////////////////////////////////////////////////////////
  // 处理描述子归一化选项
  //////////////////////////////////////////////////////////////////////////////
  
  // 将归一化方式字符串转换为小写，便于比较
  StringToLower(&descriptor_normalization);
  
  // 根据用户选择设置SIFT提取器的归一化方式
  if (descriptor_normalization == "l1_root") {
    // L1_ROOT归一化：先进行L1归一化，再开平方根
    // 这种归一化方式对光照变化更鲁棒
    options.sift_extraction->normalization =
      SiftExtractionOptions::Normalization::L1_ROOT;
  } else if (descriptor_normalization == "l2") {
    // L2归一化：标准的欧几里得范数归一化
    // 这是最常用的归一化方式
    options.sift_extraction->normalization =
      SiftExtractionOptions::Normalization::L2;
  } else {
    // 如果用户输入了无效的归一化方式，输出错误信息并退出
    std::cerr << "ERROR: Invalid `descriptor_normalization`"
              << std::endl;
    return EXIT_FAILURE;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 处理图像列表文件
  //////////////////////////////////////////////////////////////////////////////
  
  // 如果用户指定了图像列表文件路径
  if (!image_list_path.empty()) {
    // 读取图像列表文件，获取要处理的图像文件名列表
    reader_options.image_list = ReadTextFileLines(image_list_path);
    
    // 如果图像列表为空，直接返回成功（没有图像需要处理）
    if (reader_options.image_list.empty()) {
      return EXIT_SUCCESS;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // 验证相机模型和参数
  //////////////////////////////////////////////////////////////////////////////
  
  // 检查指定的相机模型是否存在
  if (!ExistsCameraModelWithName(reader_options.camera_model)) {
    std::cerr << "ERROR: Camera model does not exist" << std::endl;
  }

  // 验证相机参数是否与相机模型兼容
  if (!VerifyCameraParams(reader_options.camera_model,
                          reader_options.camera_params)) {
    return EXIT_FAILURE;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 验证GPU相关参数
  //////////////////////////////////////////////////////////////////////////////
  
  // 如果使用GPU，验证GPU参数的有效性
  if (!VerifySiftGPUParams(options.sift_extraction->use_gpu)) {
    return EXIT_FAILURE;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 创建Qt应用程序（GPU模式需要）
  //////////////////////////////////////////////////////////////////////////////
  
  std::unique_ptr<QApplication> app;
  
  // 如果使用GPU且支持OpenGL，需要创建Qt应用程序
  // 这是因为GPU版本的SIFT提取器需要OpenGL上下文
  if (options.sift_extraction->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }
  
  //////////////////////////////////////////////////////////////////////////////
  // 创建和启动特征提取器
  //////////////////////////////////////////////////////////////////////////////
  
  // 创建SIFT特征提取器，传入图像读取选项和SIFT提取选项
  SiftFeatureExtractor feature_extractor(reader_options,
                                         *options.sift_extraction);

  // 根据计算模式选择不同的执行方式
  if (options.sift_extraction->use_gpu && kUseOpenGL) {
    // GPU模式：在OpenGL上下文中运行特征提取器
    // 这确保了GPU计算资源的正确管理
    RunThreadWithOpenGLContext(&feature_extractor);
  } else {
    // CPU模式：直接启动特征提取器并等待完成
    feature_extractor.Start();
    feature_extractor.Wait();
  }

  // 程序执行成功
  return EXIT_SUCCESS;
}

/**
 * 运行特征导入器的主函数
 * 
 * 该函数负责从外部文件导入预计算的特征点数据到COLMAP数据库，
 * 而不是重新提取特征。适用于用户已有特征文件或使用其他工具提取特征的场景
 * 
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return 程序执行状态码
 */
int RunFeatureImporter(int argc, char** argv) {
  // 声明局部变量
  std::string import_path;           // 特征文件导入路径
  std::string image_list_path;       // 图像列表文件路径
  int camera_mode = -1;              // 相机模式（-1表示未指定）

  //////////////////////////////////////////////////////////////////////////////
  // 创建和配置选项管理器
  //////////////////////////////////////////////////////////////////////////////
  
  OptionManager options;  // 创建选项管理器对象
  
  // 添加各种配置选项组
  options.AddDatabaseOptions();        // 数据库相关选项（数据库路径等）
  options.AddImageOptions();           // 图像相关选项（图像路径、相机模型等）
  options.AddDefaultOption("camera_mode", &camera_mode);  // 相机模式选项
  options.AddRequiredOption("import_path", &import_path); // 必需的特征导入路径选项
  options.AddDefaultOption("image_list_path", &image_list_path);  // 图像列表路径选项
  options.AddExtractionOptions();      // 特征提取相关选项（用于验证和配置）
  options.Parse(argc, argv);          // 解析命令行参数

  //////////////////////////////////////////////////////////////////////////////
  // 配置图像读取器选项
  //////////////////////////////////////////////////////////////////////////////
  
  // 获取图像读取器选项的副本
  ImageReaderOptions reader_options = *options.image_reader;
  
  // 设置数据库路径和图像路径
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  // 如果指定了相机模式，则根据相机模式更新读取器选项
  if (camera_mode >= 0) {
    UpdateImageReaderOptionsFromCameraMode(reader_options,
                                           (CameraMode)camera_mode);
  }

  //////////////////////////////////////////////////////////////////////////////
  // 处理图像列表文件
  //////////////////////////////////////////////////////////////////////////////
  
  // 如果用户指定了图像列表文件路径
  if (!image_list_path.empty()) {
    // 读取图像列表文件，获取要处理的图像文件名列表
    reader_options.image_list = ReadTextFileLines(image_list_path);
    
    // 如果图像列表为空，直接返回成功（没有图像需要处理）
    if (reader_options.image_list.empty()) {
      return EXIT_SUCCESS;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // 验证相机参数
  //////////////////////////////////////////////////////////////////////////////
  
  // 验证相机参数是否与相机模型兼容
  // 这是必要的，因为导入的特征需要正确的相机参数进行后续处理
  if (!VerifyCameraParams(reader_options.camera_model,
                          reader_options.camera_params)) {
    return EXIT_FAILURE;  // 相机参数验证失败，退出程序
  }

  //////////////////////////////////////////////////////////////////////////////
  // 创建和启动特征导入器
  //////////////////////////////////////////////////////////////////////////////
  
  // 创建特征导入器，传入图像读取选项和特征文件导入路径
  // FeatureImporter负责将外部特征文件转换为COLMAP数据库格式
  FeatureImporter feature_importer(reader_options, import_path);
  
  // 启动特征导入过程
  feature_importer.Start();
  
  // 等待导入过程完成
  feature_importer.Wait();

  // 程序执行成功
  return EXIT_SUCCESS;
}

/**
 * 运行穷举特征匹配器的主函数
 * 
 * 该函数负责执行穷举式特征匹配，即计算所有图像对之间的特征点匹配关系。
 * 穷举匹配适用于图像数量较少或需要完整匹配信息的场景，但计算复杂度为O(n²)
 * 
 * @param argc 命令行参数数量
 * @argv 命令行参数数组
 * @return 程序执行状态码
 */
int RunExhaustiveMatcher(int argc, char** argv) {
  //////////////////////////////////////////////////////////////////////////////
  // 创建和配置选项管理器
  //////////////////////////////////////////////////////////////////////////////
  
  OptionManager options;  // 创建选项管理器对象
  
  // 添加必要的配置选项组
  options.AddDatabaseOptions();              // 数据库相关选项（数据库路径等）
  options.AddExhaustiveMatchingOptions();   // 穷举匹配相关选项（匹配策略、阈值等）
  options.Parse(argc, argv);                // 解析命令行参数

  //////////////////////////////////////////////////////////////////////////////
  // 验证GPU相关参数
  //////////////////////////////////////////////////////////////////////////////
  
  // 验证SIFT匹配的GPU参数是否有效
  // 这确保GPU配置与系统硬件和驱动兼容
  if (!VerifySiftGPUParams(options.sift_matching->use_gpu)) {
    return EXIT_FAILURE;  // GPU参数验证失败，退出程序
  }

  //////////////////////////////////////////////////////////////////////////////
  // 创建Qt应用程序（GPU模式需要）
  //////////////////////////////////////////////////////////////////////////////
  
  std::unique_ptr<QApplication> app;  // Qt应用程序智能指针
  
  // 如果使用GPU且支持OpenGL，需要创建Qt应用程序
  // 这是因为GPU版本的SIFT匹配器需要OpenGL上下文来管理GPU资源
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }
  
  //////////////////////////////////////////////////////////////////////////////
  // 创建穷举特征匹配器
  //////////////////////////////////////////////////////////////////////////////
  
  // 创建穷举特征匹配器实例，传入三个关键参数：
  // 1. exhaustive_matching: 穷举匹配的配置选项（如匹配策略、过滤条件等）
  // 2. sift_matching: SIFT匹配算法的具体参数（如距离阈值、比率测试等）
  // 3. database_path: 数据库路径，用于读取特征点和存储匹配结果
  ExhaustiveFeatureMatcher feature_matcher(*options.exhaustive_matching,
                                           *options.sift_matching,
                                           *options.database_path);

  //////////////////////////////////////////////////////////////////////////////
  // 执行特征匹配
  //////////////////////////////////////////////////////////////////////////////
  
  // 根据计算模式选择不同的执行方式
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    // GPU模式：在OpenGL上下文中运行特征匹配器
    // 这确保了GPU计算资源的正确管理和OpenGL上下文的可用性
    // GPU模式通常能显著加速匹配过程，特别是对于大量特征点的情况
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    // CPU模式：直接启动特征匹配器并等待完成
    // CPU模式虽然速度较慢，但兼容性更好，不依赖GPU硬件
    feature_matcher.Start();    // 启动匹配过程
    feature_matcher.Wait();     // 等待所有匹配任务完成
  }

  // 程序执行成功
  return EXIT_SUCCESS;
}

int RunMatchesImporter(int argc, char** argv) {
  std::string match_list_path;
  std::string match_type = "pairs";

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("match_list_path", &match_list_path);
  options.AddDefaultOption("match_type", &match_type,
                           "{'pairs', 'raw', 'inliers'}");
  options.AddMatchingOptions();
  options.Parse(argc, argv);

  if (!VerifySiftGPUParams(options.sift_matching->use_gpu)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  std::unique_ptr<Thread> feature_matcher;
  if (match_type == "pairs") {
    ImagePairsMatchingOptions matcher_options;
    matcher_options.match_list_path = match_list_path;
    feature_matcher.reset(new ImagePairsFeatureMatcher(
        matcher_options, *options.sift_matching, *options.database_path));
  } else if (match_type == "raw" || match_type == "inliers") {
    FeaturePairsMatchingOptions matcher_options;
    matcher_options.match_list_path = match_list_path;
    matcher_options.verify_matches = match_type == "raw";
    feature_matcher.reset(new FeaturePairsFeatureMatcher(
        matcher_options, *options.sift_matching, *options.database_path));
  } else {
    std::cerr << "ERROR: Invalid `match_type`";
    return EXIT_FAILURE;
  }

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(feature_matcher.get());
  } else {
    feature_matcher->Start();
    feature_matcher->Wait();
  }

  return EXIT_SUCCESS;
}

int RunSequentialMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddSequentialMatchingOptions();
  options.Parse(argc, argv);

  if (!VerifySiftGPUParams(options.sift_matching->use_gpu)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  SequentialFeatureMatcher feature_matcher(*options.sequential_matching,
                                           *options.sift_matching,
                                           *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

  return EXIT_SUCCESS;
}

int RunSpatialMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddSpatialMatchingOptions();
  options.Parse(argc, argv);

  if (!VerifySiftGPUParams(options.sift_matching->use_gpu)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  SpatialFeatureMatcher feature_matcher(*options.spatial_matching,
                                        *options.sift_matching,
                                        *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

  return EXIT_SUCCESS;
}

int RunTransitiveMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddTransitiveMatchingOptions();
  options.Parse(argc, argv);

  if (!VerifySiftGPUParams(options.sift_matching->use_gpu)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  TransitiveFeatureMatcher feature_matcher(*options.transitive_matching,
                                           *options.sift_matching,
                                           *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

  return EXIT_SUCCESS;
}

int RunVocabTreeMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddVocabTreeMatchingOptions();
  options.Parse(argc, argv);

  if (!VerifySiftGPUParams(options.sift_matching->use_gpu)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  VocabTreeFeatureMatcher feature_matcher(*options.vocab_tree_matching,
                                          *options.sift_matching,
                                          *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

  return EXIT_SUCCESS;
}

}  // namespace colmap
