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

#include "exe/sfm.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "base/reconstruction.h"
#include "controllers/automatic_reconstruction.h"
#include "controllers/bundle_adjustment.h"
#include "controllers/hierarchical_mapper.h"
#include "exe/gui.h"
#include "util/misc.h"
#include "util/opengl_utils.h"
#include "util/option_manager.h"

#include "feature/sift.h"           // 包含SiftExtractionOptions和SiftMatchingOptions的完整定义
#include "feature/extraction.h"     // 包含SiftFeatureExtractor的完整定义
#include "feature/matching.h"       // 包含ExhaustiveFeatureMatcher的完整定义
#include "controllers/incremental_mapper.h" // 包含IncrementalMapperController的完整定义
#include "util/progress_bar.h"     // 包含进度条功能

#include "base/undistortion.h"

#include "util/logging.h"
#include <mutex>

#include <yaml-cpp/yaml.h>
#include <fstream>

namespace colmap {

/**
 * [功能描述]：验证相机参数的有效性
 * @param camera_model：相机模型名称，如"PINHOLE"、"SIMPLE_PINHOLE"等
 * @param params：相机参数字符串，以逗号分隔的数值
 * @return bool：返回true表示参数有效，false表示参数无效
 */
bool VerifyCameraParams(const std::string& camera_model,
                        const std::string& params) {
  // 检查相机模型名称是否存在
  if (!ExistsCameraModelWithName(camera_model)) {
    std::cerr << "ERROR: Camera model does not exist" << std::endl;
    return false;
  }

  // 将参数字符串转换为double类型的向量
  const std::vector<double> camera_params = CSVToVector<double>(params);
  // 根据相机模型名称获取对应的模型ID
  const int camera_model_id = CameraModelNameToId(camera_model);

  // 如果提供了相机参数，则验证参数的有效性
  if (camera_params.size() > 0 &&
      !CameraModelVerifyParams(camera_model_id, camera_params)) {
    std::cerr << "Error: Invalid camera parameters" << std::endl;
    return false;
  }
  if (camera_params.size() == 0) {
    std::cerr << "错误: 相机参数为空" << std::endl;
    return false;
  }
  return true;  // 所有验证都通过，返回true
}

/**
 * [功能描述]：验证SIFT GPU参数的有效性
 * @param use_gpu：是否使用GPU加速SIFT特征提取/匹配
 * @return bool：返回true表示可以使用GPU，false表示不能使用GPU
 */
bool VerifySiftGPUParams(const bool use_gpu) {
  // 检查是否启用了CUDA或OpenGL支持
  // 如果没有启用这些GPU支持，则不能使用SIFT GPU功能
#if !defined(CUDA_ENABLED) && !defined(OPENGL_ENABLED)
  if (use_gpu) {
    std::cerr << "ERROR: Cannot use Sift GPU without CUDA or OpenGL support; "
                 "set SiftExtraction.use_gpu or SiftMatching.use_gpu to false."
              << std::endl;
    return false;
  }
#endif
  return true;  // 有GPU支持或不需要GPU，返回true
}

/**
 * [功能描述]：从YAML配置文件运行完整的重建流程，包括特征提取、特征匹配和增量重建
 * @param argc：命令行参数数量
 * @param argv：命令行参数数组
 * @return int：返回执行状态，EXIT_SUCCESS表示成功，EXIT_FAILURE表示失败
 */
int RunReconstructorFromYaml(int argc, char** argv)
{
  // 计时
  Timer timer;
  timer.Start();

  // 构建配置文件路径：项目根目录下的config文件夹
  std::string config_file = "/home/goslam/catkin_colmap-pcd/src/colmap-pcd/config/reconstruction_config.yaml";

  // 检查配置文件是否存在
  if (!ExistsFile(config_file)) {
    std::cout << "配置文件不存在: " << config_file << std::endl;
    return EXIT_FAILURE;
  } else {
    std::cout << "配置文件路径: " << config_file << std::endl;
  }

  // 加载YAML文件
  YAML::Node config = YAML::LoadFile(config_file);
  
  // 创建选项管理器
  OptionManager options;
  options.AddAllOptions();
  
  // 从YAML读取基本路径配置
  std::string workspace_path, image_path, database_path;
  if (config["workspace_path"]) {
    workspace_path = config["workspace_path"].as<std::string>();
  } else {
    std::cerr << "ERROR: 配置文件中缺少workspace_path参数" << std::endl;
    return EXIT_FAILURE;
  }
  
  if (config["image_path"]) {
    image_path = config["image_path"].as<std::string>();
  } else {
    std::cerr << "ERROR: 配置文件中缺少image_path参数" << std::endl;
    return EXIT_FAILURE;
  }
  
  // 设置数据库路径
  database_path = JoinPaths(workspace_path, "database.db");
  
  // 设置基本路径
  *options.database_path = database_path;
  *options.image_path = image_path;

  // TODO: 读取点云文件，读取相机先验位姿
  if (config["lidar_pointcloud_path"]) {
    std::string lidar_pointcloud_path = config["lidar_pointcloud_path"].as<std::string>();
    if (!lidar_pointcloud_path.empty()) {
      options.mapper->if_add_lidar_constraint = true;
      options.mapper->lidar_pointcloud_path = lidar_pointcloud_path;
    }
  }

  // 从YAML读取并设置其他参数
  if (config["mask_path"]) {
    std::string mask_path = config["mask_path"].as<std::string>();
    if (!mask_path.empty()) {
      options.image_reader->mask_path = mask_path;
    }
  }
  
  if (config["camera_model"]) {
    options.image_reader->camera_model = config["camera_model"].as<std::string>();
  }

  if (config["camera_params"]) {
    options.image_reader->camera_params = config["camera_params"].as<std::string>();
  }
  
  if (config["single_camera"]) {
    options.image_reader->single_camera = config["single_camera"].as<bool>();
  }
  
  if (config["use_gpu"]) {
    bool use_gpu = config["use_gpu"].as<bool>();
    options.sift_extraction->use_gpu = use_gpu;
    options.sift_matching->use_gpu = use_gpu;
  }
  
  // if (config["num_threads"]) {
  //   int num_threads = config["num_threads"].as<int>();
  //   options.sift_extraction->num_threads = num_threads;
  //   options.sift_matching->num_threads = num_threads;
  //   options.mapper->num_threads = num_threads;
  // }
  
  // if (config["gpu_index"]) {
  //   std::string gpu_index = config["gpu_index"].as<std::string>();
  //   options.sift_extraction->gpu_index = gpu_index;
  //   options.sift_matching->gpu_index = gpu_index;
  // }
  
  // // 根据数据类型和质量调整配置
  // if (config["data_type"]) {
  //   std::string data_type = config["data_type"].as<std::string>();
  //   StringToLower(&data_type);
  //   if (data_type == "video") {
  //     options.ModifyForVideoData();
  //   } else if (data_type == "individual") {
  //     options.ModifyForIndividualData();
  //   } else if (data_type == "internet") {
  //     options.ModifyForInternetData();
  //   }
  // }
  
  // if (config["quality"]) {
  //   std::string quality = config["quality"].as<std::string>();
  //   StringToLower(&quality);
  //   if (quality == "low") {
  //     options.ModifyForLowQuality();
  //   } else if (quality == "medium") {
  //     options.ModifyForMediumQuality();
  //   } else if (quality == "high") {
  //     options.ModifyForHighQuality();
  //   } else if (quality == "extreme") {
  //     options.ModifyForExtremeQuality();
  //   }
  // }

  // // SIFT特征提取参数
  // options.sift_extraction->max_image_size = config["sift_extraction"]["max_image_size"].as<int>();
  // options.sift_extraction->max_num_features = config["sift_extraction"]["max_num_features"].as<int>();
  // options.sift_extraction->first_octave = config["sift_extraction"]["first_octave"].as<int>();
  // options.sift_extraction->num_octaves = config["sift_extraction"]["num_octaves"].as<int>();
  // options.sift_extraction->octave_resolution = config["sift_extraction"]["octave_resolution"].as<int>();
  // options.sift_extraction->peak_threshold = config["sift_extraction"]["peak_threshold"].as<double>();
  // options.sift_extraction->edge_threshold = config["sift_extraction"]["edge_threshold"].as<double>();
  // options.sift_extraction->estimate_affine_shape = config["sift_extraction"]["estimate_affine_shape"].as<bool>();
  // options.sift_extraction->max_num_orientations = config["sift_extraction"]["max_num_orientations"].as<int>();
  // options.sift_extraction->upright = config["sift_extraction"]["upright"].as<bool>();
  // options.sift_extraction->domain_size_pooling = config["sift_extraction"]["domain_size_pooling"].as<bool>();

  // // SIFT特征匹配参数
  // options.sift_matching->max_ratio = config["sift_matching"]["max_ratio"].as<double>();
  // options.sift_matching->max_distance = config["sift_matching"]["max_distance"].as<double>();
  // options.sift_matching->cross_check = config["sift_matching"]["cross_check"].as<bool>();
  // options.sift_matching->max_num_matches = config["sift_matching"]["max_num_matches"].as<int>();
  // options.sift_matching->max_error = config["sift_matching"]["max_error"].as<double>();
  // options.sift_matching->confidence = config["sift_matching"]["confidence"].as<double>();
  // options.sift_matching->min_num_trials = config["sift_matching"]["min_num_trials"].as<int>();
  // options.sift_matching->max_num_trials = config["sift_matching"]["max_num_trials"].as<int>();
  // options.sift_matching->min_inlier_ratio = config["sift_matching"]["min_inlier_ratio"].as<double>();
  // options.sift_matching->min_num_inliers = config["sift_matching"]["min_num_inliers"].as<int>();
  // options.sift_matching->multiple_models = config["sift_matching"]["multiple_models"].as<bool>();
  // options.sift_matching->guided_matching = config["sift_matching"]["guided_matching"].as<bool>();
  // options.sift_matching->planar_scene = config["sift_matching"]["planar_scene"].as<bool>();

  // 映射器参数
  // options.mapper->min_track_length = config["mapper"]["min_track_length"].as<int>();
  // options.mapper->max_track_length = config["mapper"]["max_track_length"].as<int>();
  // options.mapper->min_focal_length_ratio = config["mapper"]["min_focal_length_ratio"].as<double>();
  // options.mapper->max_focal_length_ratio = config["mapper"]["max_focal_length_ratio"].as<double>();
  // options.mapper->max_extra_param = config["mapper"]["max_extra_param"].as<double>();
  // options.mapper->min_num_matches = config["mapper"]["min_num_matches"].as<int>();
  // options.mapper->init_min_num_inliers = config["mapper"]["init_min_num_inliers"].as<int>();
  // options.mapper->init_min_triangulation_angle = config["mapper"]["init_min_triangulation_angle"].as<double>();
  // options.mapper->init_max_reg_trials = config["mapper"]["init_max_reg_trials"].as<int>();
  // options.mapper->abs_pose_max_error = config["mapper"]["abs_pose_max_error"].as<double>();
  // options.mapper->abs_pose_min_num_inliers = config["mapper"]["abs_pose_min_num_inliers"].as<int>();
  // options.mapper->abs_pose_min_triangulation_angle = config["mapper"]["abs_pose_min_triangulation_angle"].as<double>();
  // options.mapper->filter_max_reproj_error = config["mapper"]["filter_max_reproj_error"].as<double>();
  // options.mapper->filter_min_track_length = config["mapper"]["filter_min_track_length"].as<int>();
  // options.mapper->filter_min_triangulation_angle = config["mapper"]["filter_min_triangulation_angle"].as<double>();

  // 验证配置
  if (!options.Check()) {
    std::cerr << "ERROR: 配置验证失败" << std::endl;
    return EXIT_FAILURE;
  }
  
  // 检查工作空间和图像目录
  if (!ExistsDir(workspace_path)) {
    std::cout << "工作空间目录不存在，正在创建: " << workspace_path << std::endl;
    CreateDirIfNotExists(workspace_path);
  }
  
  if (!ExistsDir(image_path)) {
    std::cerr << "ERROR: 图像目录不存在: " << image_path << std::endl;
    return EXIT_FAILURE;
  }
  
  std::cout << "=== 开始执行重建流程 ===" << std::endl;
  
  // 第一步：特征提取
  std::cout << "步骤1: 特征提取..." << std::endl;
  {
    // 配置图像读取器选项
    ImageReaderOptions reader_options = *options.image_reader;
    reader_options.database_path = database_path;
    reader_options.image_path = image_path;

    std::string descriptor_normalization = "l1_root";
    options.sift_extraction->normalization = SiftExtractionOptions::Normalization::L1_ROOT;
    
    // 验证相机参数
    if (!VerifyCameraParams(reader_options.camera_model,
                            reader_options.camera_params)) {
      std::cerr << "ERROR: 相机参数验证失败" << std::endl;
      return EXIT_FAILURE;
    }
    
    // 验证GPU参数
    if (!VerifySiftGPUParams(options.sift_extraction->use_gpu)) {
      std::cerr << "ERROR: GPU参数验证失败" << std::endl;
      return EXIT_FAILURE;
    }

    // 创建特征提取器
    SiftFeatureExtractor feature_extractor(reader_options, *options.sift_extraction);

    std::cout << "图像列表: " << reader_options.image_list.size() << std::endl;

    // 执行特征提取
    if (options.sift_extraction->use_gpu && kUseOpenGL) {
      // GPU模式
      std::unique_ptr<QApplication> app(new QApplication(argc, argv));
      RunThreadWithOpenGLContext(&feature_extractor);
    } else {
      // CPU模式
      feature_extractor.Start();
      feature_extractor.Wait();
    }
    
    std::cout << "特征提取完成" << std::endl;
  }
  
  // 第二步：特征匹配
  std::cout << "步骤2: 特征匹配..." << std::endl;
  {
    // 验证GPU参数
    if (!VerifySiftGPUParams(options.sift_matching->use_gpu)) {
      std::cerr << "ERROR: GPU参数验证失败" << std::endl;
      return EXIT_FAILURE;
    }
    
    // 创建穷举特征匹配器
    // ExhaustiveFeatureMatcher feature_matcher(*options.exhaustive_matching,
    SequentialFeatureMatcher feature_matcher(*options.sequential_matching,
                                             *options.sift_matching,
                                             database_path);
    
    // 执行特征匹配
    if (options.sift_matching->use_gpu && kUseOpenGL) {
      // GPU模式
      std::unique_ptr<QApplication> app(new QApplication(argc, argv));
      RunThreadWithOpenGLContext(&feature_matcher);
    } else {
      // CPU模式
      feature_matcher.Start();
      feature_matcher.Wait();
    }
    
    std::cout << "特征匹配完成" << std::endl;
  }
  
  // 第三步：增量重建
  std::cout << "步骤3: 增量重建..." << std::endl;
  // 创建重建管理器
  ReconstructionManager reconstruction_manager;
  {
    
    // 创建增量映射器控制器
    IncrementalMapperController mapper(options.mapper.get(), image_path,
                                       database_path, &reconstruction_manager);
    
    // 执行增量重建
    mapper.Start();
    mapper.Wait();
    
    // 检查重建结果
    if (reconstruction_manager.Size() == 0) {
      std::cerr << "ERROR: Reconstruction failed, no sparse model generated" << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "重建图像数量: " << reconstruction_manager.Get(0).NumRegImages() << std::endl;
    
    // 保存重建结果
    // const std::string sparse_path = JoinPaths(workspace_path, "sparse");
    // CreateDirIfNotExists(sparse_path);
    // reconstruction_manager.Get(0).Write(sparse_path);
    
    // // 保存项目配置文件
    // options.Write(JoinPaths(sparse_path, "project.ini"));
    
    std::cout << "增量重建完成" << std::endl;
  }

  // 第四步：全局BA
  std::cout << "步骤4: 全局BA..." << std::endl;
  if (false)
  {
    std::string input_path;
    std::string output_path;
    output_path = JoinPaths(workspace_path, "sparse");

    if (!ExistsDir(output_path)) {
      std::cout << "输出文件不存在: " << output_path << std::endl;
      return EXIT_FAILURE;
    }

    // Reconstruction reconstruction;
    // reconstruction.Read(input_path);

    BundleAdjustmentController ba_controller(options, &reconstruction_manager.Get(0));
    ba_controller.Start();
    ba_controller.Wait();

    // reconstruction_manager.Get(0).Write(output_path);

    std::cout << "全局BA完成" << std::endl;
  }

  // 第五步：保存结果
  std::cout << "步骤5: 保存结果..." << std::endl;
  {
    std::string output_path;
    output_path = JoinPaths(workspace_path, "colmap");
    if (!ExistsDir(output_path)) {
      CreateDirIfNotExists(output_path);
    }

    UndistortCameraOptions undistortion_options;
    COLMAPUndistorter undistorter(undistortion_options,
                                  reconstruction_manager.Get(0),
                                  *options.image_path, output_path);

    undistorter.Start();
    undistorter.Wait();

    // 保存稀疏模型
    std::string sparse_path;
    sparse_path = workspace_path + "/sparse/0";
    // if txt
    // reconstruction_manager.Get(0).WriteText(output_path);
    // if bin
    // reconstruction_manager.Get(0).WriteBinary(sparse_path);

    // 将去畸变后的重建结果写入sparse目录
    // undistorted_reconstruction.Write(sparse_path);

    std::cout << "结果保存完成" << std::endl;
  }

  // 生成配置文件
  // std::cout << "生成配置文件..." << std::endl;
  // GenerateDefaultConfigFile(workspace_path);
  
  std::cout << "=== 重建流程完成 ===" << std::endl;
  timer.PrintMinutes();
  return EXIT_SUCCESS;
}

// 全局进度管理器指针和进度状态
static MultiStageProgressManager* g_progress_manager = nullptr;
static std::mutex g_progress_mutex;
static double g_last_progress = 0.0;  // 保存最后的进度值
static bool g_is_running = false;     // 重建是否正在运行

// 获取重建进度的函数
double GetReconstructionProgress() {
  std::lock_guard<std::mutex> lock(g_progress_mutex);
  if (g_progress_manager) {
    g_last_progress = g_progress_manager->GetOverallProgress();
    return g_last_progress;
  }
  // 如果没有活跃的进度管理器，返回上次保存的进度值
  return g_last_progress;
}

// 检查重建是否正在运行
bool IsReconstructionRunning() {
  std::lock_guard<std::mutex> lock(g_progress_mutex);
  return g_is_running;
}

// 重置进度状态（开始新的重建前调用）
void ResetReconstructionProgress() {
  std::lock_guard<std::mutex> lock(g_progress_mutex);
  g_last_progress = 0.0;
  g_is_running = false;
  g_progress_manager = nullptr;
}

int AutomaticReconstructor(std::string _workspace_path, ReconstructionProgressCallback callback, bool use_gpu) {
  static std::once_flag glog_once;
  std::call_once(glog_once, []() {
    static char arg0[] = "colmap_api";
    char* argv[] = {arg0, nullptr};
    InitializeGlog(argv);   // 会自动把 FLAGS_minloglevel 设成 3
  });

  // 计时
  Timer timer;
  timer.Start();

  // 创建多阶段进度管理器
  std::vector<std::string> stage_names = {
    "特征提取",
    "特征匹配", 
    "增量重建"
  };
  std::vector<double> stage_weights = {0.2, 0.3, 0.5}; // 各阶段相对耗时权重
  MultiStageProgressManager progress_manager(stage_names, stage_weights);

  // 设置全局指针和运行状态
  {
    std::lock_guard<std::mutex> lock(g_progress_mutex);
    g_progress_manager = &progress_manager;
    g_is_running = true;
    g_last_progress = 0.0;
  }

  // 创建选项管理器
  OptionManager options;
  options.AddAllOptions();
  
  // 从YAML读取基本路径配置
  std::string workspace_path, image_path, database_path;
  if (!_workspace_path.empty()) {
    workspace_path = _workspace_path;
  } else {
    std::cerr << "ERROR: workspace_path is empty" << std::endl;
    return EXIT_FAILURE;
  }
  
  image_path = JoinPaths(workspace_path, "images");
  if (!ExistsDir(image_path)) {
    std::cerr << "ERROR: images directory not found:" << image_path << std::endl;
    return EXIT_FAILURE;
  }

  // 设置数据库路径
  database_path = JoinPaths(workspace_path, "database.db");
  
  // 设置基本路径
  *options.database_path = database_path;
  *options.image_path = image_path;

  // TODO: 读取点云文件，读取相机先验位姿
  std::string lidar_pointcloud_path = JoinPaths(workspace_path, "plane_cloud.ply");
  if (ExistsFile(lidar_pointcloud_path)) {
    options.mapper->if_add_lidar_constraint = true;
    options.mapper->lidar_pointcloud_path = lidar_pointcloud_path;
  }

  options.image_reader->camera_model = "PINHOLE";

  // 从 cameras.txt 读取相机内参
  std::string cameras_file_path = JoinPaths(workspace_path, "sparse/0/cameras.txt");
  if (ExistsFile(cameras_file_path)) {
    std::ifstream cameras_file(cameras_file_path);
    std::string line;
    
    // 逐行读取文件，跳过注释行
    while (std::getline(cameras_file, line)) {
      // 跳过注释行和空行
      if (line.empty() || line[0] == '#') {
        continue;
      }
      
      // 解析相机参数行
      std::istringstream iss(line);
      int camera_id;
      std::string model;
      int width, height;
      double fx, fy, cx, cy;
      
      // 按顺序读取：CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
      if (iss >> camera_id >> model >> width >> height >> fx >> fy >> cx >> cy) {
        // 构建相机参数字符串，格式为 "fx,fy,cx,cy"
        std::ostringstream params_stream;
        params_stream << fx << "," << fy << "," << cx << "," << cy;
        options.image_reader->camera_params = params_stream.str();
        
        std::cout << "从cameras.txt读取相机内参: fx=" << fx 
                  << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << std::endl;
        break; // 只读取第一个相机的参数
      } else {
        std::cerr << "WARNING: 无法解析cameras.txt中的相机参数行: " << line << std::endl;
      }
    }
    cameras_file.close();
  } else {
    std::cerr << "WARNING: 未找到cameras.txt文件" << std::endl;
  }

  // 从 init_pose.txt 文件中读取初始位姿
  std::string init_pose_file_path = JoinPaths(workspace_path, "init_pose.txt");
  if (ExistsFile(init_pose_file_path)) {
    std::ifstream init_pose_file(init_pose_file_path);
    std::string line;
    
    // 逐行读取文件，跳过注释行
    while (std::getline(init_pose_file, line)) {
      // 跳过注释行和空行
      if (line.empty() || line[0] == '#') {
        continue;
      }
      
      // 解析初始位姿参数行
      std::istringstream iss(line);
      double x, y, z, roll, pitch, yaw;
      
      // 按顺序读取：x y z roll pitch yaw
      if (iss >> x >> y >> z >> roll >> pitch >> yaw) {
        options.mapper->init_image_x = x;
        options.mapper->init_image_y = y;
        options.mapper->init_image_z = z;
        options.mapper->init_image_roll = roll;
        options.mapper->init_image_pitch = pitch;
        options.mapper->init_image_yaw = yaw;
      }
    }
    init_pose_file.close();
  } else {
    std::cerr << "WARNING: 未找到cameras.txt文件" << std::endl;
  }
  
  options.image_reader->single_camera = true;

  options.sift_extraction->use_gpu = false;
  options.sift_matching->use_gpu = false;
  
  if (use_gpu) {
    options.sift_extraction->use_gpu = true;
    options.sift_matching->use_gpu = true;

    // GPU模式 - 创建虚假的argc和argv参数
    static char app_name[] = "colmap_api";
    static char platform_arg[] = "-platform";
    static char platform_val[] = "offscreen";  
    static char* argv[] = {app_name, platform_arg, platform_val, nullptr};
    static int argc = 3;
    
    // 检查是否已经存在QApplication实例
    // 如果桌面端软件已经创建了QApplication，就不需要再创建新的
    std::unique_ptr<QApplication> app;
    if (!QApplication::instance()) {
      // 只有在不存在QApplication实例时才创建新的
      std::cout << "创建QApplication实例..." << std::endl;
      app.reset(new QApplication(argc, argv));

      // 检测屏幕可用性，如果没有屏幕就降级到CPU
      if (QGuiApplication::screens().isEmpty()) {
        std::cout << "警告: 无可用屏幕，GPU模式可能无法工作，建议使用CPU模式" << std::endl;
      }
    }
  }
  
  // 验证配置
  if (!options.Check()) {
    std::cerr << "ERROR: 配置验证失败" << std::endl;
    return EXIT_FAILURE;
  }
  
  // 检查工作空间和图像目录
  if (!ExistsDir(workspace_path)) {
    std::cout << "工作空间目录不存在，正在创建: " << workspace_path << std::endl;
    CreateDirIfNotExists(workspace_path);
  }
  
  if (!ExistsDir(image_path)) {
    std::cerr << "ERROR: 图像目录不存在: " << image_path << std::endl;
    return EXIT_FAILURE;
  }
  
  // 第一步：特征提取
  // progress_manager.StartStage(0, 0); // 总数将由线程自动设置
  {
    // 配置图像读取器选项
    ImageReaderOptions reader_options = *options.image_reader;
    reader_options.database_path = database_path;
    reader_options.image_path = image_path;

    std::string descriptor_normalization = "l1_root";
    options.sift_extraction->normalization = SiftExtractionOptions::Normalization::L1_ROOT;
    
    // 验证相机参数
    if (!VerifyCameraParams(reader_options.camera_model,
                            reader_options.camera_params)) {
      std::cerr << "ERROR: 相机参数验证失败" << std::endl;
      return EXIT_FAILURE;
    }
    
    // 验证GPU参数
    if (!VerifySiftGPUParams(options.sift_extraction->use_gpu)) {
      std::cerr << "ERROR: GPU参数验证失败" << std::endl;
      return EXIT_FAILURE;
    }

    // 创建特征提取器
    SiftFeatureExtractor feature_extractor(reader_options, *options.sift_extraction);

    // 设置进度回调 - 连接特征提取器的Writer线程回调
    Database temp_database(database_path);
    ImageReader temp_reader(reader_options, &temp_database);
    const size_t total_images = temp_reader.NumImages();
    progress_manager.StartStage(0, total_images);
    
    // 设置进度回调
    size_t processed_images = 0;
    feature_extractor.AddCallback(SiftFeatureExtractor::PROGRESS_CALLBACK, [&]() {
      ++processed_images;
      progress_manager.UpdateCurrentStage(processed_images);

      if (callback != nullptr) {
        double progress = g_progress_manager->GetOverallProgress();
        callback(progress, "特征提取", false);
      }
    });

    if (use_gpu) {
      // 执行特征提取，使用OPENGL
      RunThreadWithOpenGLContext(&feature_extractor);
    } else {
      // 暂时取消使用OpenGl，直接使用CPU模式
      feature_extractor.Start();
      feature_extractor.Wait();
    }
    
    progress_manager.FinishCurrentStage();
  }
  
  // 第二步：特征匹配
  {
    // 验证GPU参数
    if (!VerifySiftGPUParams(options.sift_matching->use_gpu)) {
      std::cerr << "ERROR: GPU参数验证失败" << std::endl;
      return EXIT_FAILURE;
    }
    
    // 创建序列特征匹配器
    SequentialFeatureMatcher feature_matcher(*options.sequential_matching,
                                             *options.sift_matching,
                                             database_path);
    
    // 估算匹配任务总数（基于图像数量和重叠参数）
    Database temp_database(database_path);
    const size_t num_images = temp_database.ReadAllImages().size();
    const size_t total_matches = std::min(num_images, (size_t)options.sequential_matching->overlap);
    progress_manager.StartStage(1, num_images);
    
    // 设置进度回调
    size_t completed_matches = 0;
    feature_matcher.AddCallback(SequentialFeatureMatcher::PROGRESS_CALLBACK, [&]() {
      ++completed_matches;
      progress_manager.UpdateCurrentStage(completed_matches);

      if (callback != nullptr) {
        double progress = g_progress_manager->GetOverallProgress();
        callback(progress, "特征匹配", false);
      }
    });
    
    if (use_gpu) {
      // 执行特征匹配，使用OPENGL
      RunThreadWithOpenGLContext(&feature_matcher);
    } else {
      // 暂时取消使用OpenGl，直接使用CPU模式
      feature_matcher.Start();
      feature_matcher.Wait();
    }
    
    progress_manager.FinishCurrentStage();
  }
  
  // 第三步：增量重建
  std::cout << "步骤3: 增量重建..." << std::endl;
  // 创建重建管理器
  ReconstructionManager reconstruction_manager;
  {
    // 获取图像总数用于进度计算
    Database temp_database(database_path);
    const size_t total_images = temp_database.ReadAllImages().size();
    progress_manager.StartStage(2, total_images);
    
    // 创建增量映射器控制器
    IncrementalMapperController mapper(options.mapper.get(), image_path,
                                       database_path, &reconstruction_manager);

    // 设置进度回调函数 - 在每次图像注册成功后更新进度
    mapper.AddCallback(IncrementalMapperController::PROGRESS_CALLBACK, [&]() {
        // 获取当前已注册的图像数量
        if (reconstruction_manager.Size() > 0) {
            const size_t registered_images = reconstruction_manager.Get(0).NumRegImages();
            progress_manager.UpdateCurrentStage(registered_images, 
                                               "正在注册图像: " + std::to_string(registered_images) + "/" + std::to_string(total_images));
        }

        if (callback != nullptr) {
          double progress = g_progress_manager->GetOverallProgress();
          callback(progress, "增量重建", false);
        }
    });
    
    // 执行增量重建
    mapper.Start();
    mapper.Wait();

    // 完成当前阶段进度
    progress_manager.FinishCurrentStage();
    
    // 检查重建结果
    if (reconstruction_manager.Size() == 0) {
      std::cerr << "ERROR: Reconstruction failed, no sparse model generated" << std::endl;
      return EXIT_FAILURE;
    }

    // std::cout << "重建图像数量: " << reconstruction_manager.Get(0).NumRegImages() << std::endl;
    
    // 保存重建结果
    // const std::string sparse_path = JoinPaths(workspace_path, "sparse");
    // CreateDirIfNotExists(sparse_path);
    // reconstruction_manager.Get(0).Write(sparse_path);
    
    // // 保存项目配置文件
    // options.Write(JoinPaths(sparse_path, "project.ini"));
    
    std::cout << "增量重建完成" << std::endl;
  }

  // 第四步：全局BA
  // std::cout << "步骤4: 全局BA..." << std::endl;
  if (false)
  {
    std::string input_path;
    std::string output_path;
    output_path = JoinPaths(workspace_path, "sparse");

    if (!ExistsDir(output_path)) {
      std::cout << "输出文件不存在: " << output_path << std::endl;
      return EXIT_FAILURE;
    }

    // Reconstruction reconstruction;
    // reconstruction.Read(input_path);

    BundleAdjustmentController ba_controller(options, &reconstruction_manager.Get(0));
    ba_controller.Start();
    ba_controller.Wait();

    // reconstruction_manager.Get(0).Write(output_path);

    std::cout << "全局BA完成" << std::endl;
  }

  // 第五步：保存结果
  std::cout << "步骤4: 保存结果..." << std::endl;
  {
    std::string output_path;
    output_path = JoinPaths(workspace_path, "colmap");
    if (!ExistsDir(output_path)) {
      CreateDirIfNotExists(output_path);
    }

    UndistortCameraOptions undistortion_options;
    COLMAPUndistorter undistorter(undistortion_options,
                                  reconstruction_manager.Get(0),
                                  *options.image_path, output_path);

    undistorter.Start();
    undistorter.Wait();

    // 保存稀疏模型
    std::string sparse_path;
    sparse_path = workspace_path + "/sparse/0";
    // if txt
    // reconstruction_manager.Get(0).WriteText(output_path);
    // if bin
    // reconstruction_manager.Get(0).WriteBinary(sparse_path);

    // 将去畸变后的重建结果写入sparse目录
    // undistorted_reconstruction.Write(sparse_path);

    std::cout << "结果保存完成" << std::endl;
  }

  // 生成配置文件
  // std::cout << "生成配置文件..." << std::endl;
  // GenerateDefaultConfigFile(workspace_path);
  
  std::cout << "=== 重建流程完成 ===" << std::endl;
  timer.PrintMinutes();

  // 清除全局指针，但保留进度值
  {
    std::lock_guard<std::mutex> lock(g_progress_mutex);
    g_progress_manager = nullptr;
    g_is_running = false;
    // 不清除 g_last_progress，保持最终状态
  }

  return EXIT_SUCCESS;
}

int RunAutomaticReconstructor(int argc, char** argv) {
  AutomaticReconstructionController::Options reconstruction_options;
  std::string data_type = "individual";
  std::string quality = "high";
  std::string mesher = "poisson";

  OptionManager options;
  options.AddRequiredOption("workspace_path",
                            &reconstruction_options.workspace_path);
  options.AddRequiredOption("image_path", &reconstruction_options.image_path);
  options.AddDefaultOption("mask_path", &reconstruction_options.mask_path);
  options.AddDefaultOption("vocab_tree_path",
                           &reconstruction_options.vocab_tree_path);
  options.AddDefaultOption("data_type", &data_type,
                           "{individual, video, internet}");
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  options.AddDefaultOption("camera_model",
                           &reconstruction_options.camera_model);
  options.AddDefaultOption("single_camera",
                           &reconstruction_options.single_camera);
  options.AddDefaultOption("sparse", &reconstruction_options.sparse);
  options.AddDefaultOption("dense", &reconstruction_options.dense);
  options.AddDefaultOption("mesher", &mesher, "{poisson, delaunay}");
  options.AddDefaultOption("num_threads", &reconstruction_options.num_threads);
  options.AddDefaultOption("use_gpu", &reconstruction_options.use_gpu);
  options.AddDefaultOption("gpu_index", &reconstruction_options.gpu_index);
  options.Parse(argc, argv);

  StringToLower(&data_type);
  if (data_type == "individual") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::INDIVIDUAL;
  } else if (data_type == "video") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::VIDEO;
  } else if (data_type == "internet") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::INTERNET;
  } else {
    LOG(FATAL) << "Invalid data type provided";
  }

  StringToLower(&quality);
  if (quality == "low") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::LOW;
  } else if (quality == "medium") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::MEDIUM;
  } else if (quality == "high") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::HIGH;
  } else if (quality == "extreme") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::EXTREME;
  } else {
    LOG(FATAL) << "Invalid quality provided";
  }

  StringToLower(&mesher);
  if (mesher == "poisson") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::POISSON;
  } else if (mesher == "delaunay") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::DELAUNAY;
  } else {
    LOG(FATAL) << "Invalid mesher provided";
  }

  ReconstructionManager reconstruction_manager;

  if (reconstruction_options.use_gpu && kUseOpenGL) {
    QApplication app(argc, argv);
    AutomaticReconstructionController controller(reconstruction_options,
                                                 &reconstruction_manager);
    RunThreadWithOpenGLContext(&controller);
  } else {
    AutomaticReconstructionController controller(reconstruction_options,
                                                 &reconstruction_manager);
    controller.Start();
    controller.Wait();
  }

  return EXIT_SUCCESS;
}

int RunBundleAdjuster(int argc, char** argv) {
  LOG(ERROR)<<"Let me see what it do";
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunColorExtractor(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  reconstruction.ExtractColorsForAllImages(*options.image_path);
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunMapper(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_list_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  if (!image_list_path.empty()) {
    const auto image_names = ReadTextFileLines(image_list_path);
    options.mapper->image_names =
        std::unordered_set<std::string>(image_names.begin(), image_names.end());
  }

  ReconstructionManager reconstruction_manager;
  if (input_path != "") {
    if (!ExistsDir(input_path)) {
      std::cerr << "ERROR: `input_path` is not a directory." << std::endl;
      return EXIT_FAILURE;
    }
    reconstruction_manager.Read(input_path);
  }

  IncrementalMapperController mapper(options.mapper.get(), *options.image_path,
                                     *options.database_path,
                                     &reconstruction_manager);

  // In case a new reconstruction is started, write results of individual sub-
  // models to as their reconstruction finishes instead of writing all results
  // after all reconstructions finished.
  size_t prev_num_reconstructions = 0;
  if (input_path == "") {
    mapper.AddCallback(
        IncrementalMapperController::LAST_IMAGE_REG_CALLBACK, [&]() {
          // If the number of reconstructions has not changed, the last model
          // was discarded for some reason.
          if (reconstruction_manager.Size() > prev_num_reconstructions) {
            const std::string reconstruction_path = JoinPaths(
                output_path, std::to_string(prev_num_reconstructions));
            const auto& reconstruction =
                reconstruction_manager.Get(prev_num_reconstructions);
            CreateDirIfNotExists(reconstruction_path);
            reconstruction.Write(reconstruction_path);
            options.Write(JoinPaths(reconstruction_path, "project.ini"));
            prev_num_reconstructions = reconstruction_manager.Size();
          }
        });
  }

  mapper.Start();
  mapper.Wait();

  if (reconstruction_manager.Size() == 0) {
    std::cerr << "ERROR: failed to create sparse model" << std::endl;
    return EXIT_FAILURE;
  }

  // In case the reconstruction is continued from an existing reconstruction, do
  // not create sub-folders but directly write the results.
  if (input_path != "" && reconstruction_manager.Size() > 0) {
    reconstruction_manager.Get(0).Write(output_path);
  }

  return EXIT_SUCCESS;
}

int RunHierarchicalMapper(int argc, char** argv) {
  HierarchicalMapperController::Options hierarchical_options;
  SceneClustering::Options clustering_options;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("database_path",
                            &hierarchical_options.database_path);
  options.AddRequiredOption("image_path", &hierarchical_options.image_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("num_workers", &hierarchical_options.num_workers);
  options.AddDefaultOption("image_overlap", &clustering_options.image_overlap);
  options.AddDefaultOption("leaf_max_num_images",
                           &clustering_options.leaf_max_num_images);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  ReconstructionManager reconstruction_manager;

  HierarchicalMapperController hierarchical_mapper(
      hierarchical_options, clustering_options, *options.mapper,
      &reconstruction_manager);
  hierarchical_mapper.Start();
  hierarchical_mapper.Wait();

  if (reconstruction_manager.Size() == 0) {
    std::cerr << "ERROR: failed to create sparse model" << std::endl;
    return EXIT_FAILURE;
  }

  reconstruction_manager.Write(output_path, &options);

  return EXIT_SUCCESS;
}

int RunPointFiltering(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  size_t min_track_len = 2;
  double max_reproj_error = 4.0;
  double min_tri_angle = 1.5;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_track_len", &min_track_len);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.AddDefaultOption("min_tri_angle", &min_tri_angle);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  size_t num_filtered =
      reconstruction.FilterAllPoints3D(max_reproj_error, min_tri_angle);

  for (const auto point3D_id : reconstruction.Point3DIds()) {
    const auto& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.Track().Length() < min_track_len) {
      num_filtered += point3D.Track().Length();
      reconstruction.DeletePoint3D(point3D_id);
    }
  }

  std::cout << "Filtered observations: " << num_filtered << std::endl;

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunPointTriangulator(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  bool clear_points = false;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "clear_points", &clear_points,
      "Whether to clear all existing points and observations");
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  PrintHeading1("Loading model");

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  return RunPointTriangulatorImpl(
      reconstruction, *options.database_path, *options.image_path, output_path,
      *options.mapper, clear_points);
}

int RunPointTriangulatorImpl(Reconstruction& reconstruction,
                             const std::string database_path,
                             const std::string image_path,
                             const std::string output_path,
                             const IncrementalMapperOptions& mapper_options,
                             const bool clear_points) {
  PrintHeading1("Loading database");

  DatabaseCache database_cache;

  {
    Timer timer;
    timer.Start();

    Database database(database_path);

    const size_t min_num_matches =
        static_cast<size_t>(mapper_options.min_num_matches);
    database_cache.Load(database, min_num_matches,
                        mapper_options.ignore_watermarks,
                        mapper_options.image_names);

    if (clear_points) {
      reconstruction.DeleteAllPoints2DAndPoints3D();
      reconstruction.TranscribeImageIdsToDatabase(database);
    }

    std::cout << std::endl;
    timer.PrintMinutes();
  }

  std::cout << std::endl;

  CHECK_GE(reconstruction.NumRegImages(), 2)
      << "Need at least two images for triangulation";

  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(&reconstruction);

  //////////////////////////////////////////////////////////////////////////////
  // Triangulation
  //////////////////////////////////////////////////////////////////////////////

  const auto tri_options = mapper_options.Triangulation();

  const auto& reg_image_ids = reconstruction.RegImageIds();

  for (size_t i = 0; i < reg_image_ids.size(); ++i) {
    const image_t image_id = reg_image_ids[i];

    const auto& image = reconstruction.Image(image_id);

    PrintHeading1(StringPrintf("Triangulating image #%d (%d)", image_id, i));

    const size_t num_existing_points3D = image.NumPoints3D();

    std::cout << "  => Image sees " << num_existing_points3D << " / "
              << image.NumObservations() << " points" << std::endl;

    mapper.TriangulateImage(tri_options, image_id);

    std::cout << "  => Triangulated "
              << (image.NumPoints3D() - num_existing_points3D) << " points"
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Retriangulation
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Retriangulation");

  CompleteAndMergeTracks(mapper_options, &mapper);

  //////////////////////////////////////////////////////////////////////////////
  // Bundle adjustment
  //////////////////////////////////////////////////////////////////////////////

  auto ba_options = mapper_options.GlobalBundleAdjustment();
  ba_options.refine_focal_length = false;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = false;
  ba_options.refine_extrinsics = false;

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
    // Avoid degeneracies in bundle adjustment.
    reconstruction.FilterObservationsWithNegativeDepth();

    const size_t num_observations = reconstruction.ComputeNumObservations();

    PrintHeading1("Bundle adjustment");
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    const BundleAdjuster::OptimazePhrase phrase = BundleAdjuster::OptimazePhrase::Global;
    bundle_adjuster.SetOptimazePhrase(phrase);
    CHECK(bundle_adjuster.Solve(&reconstruction));

    size_t num_changed_observations = 0;
    num_changed_observations += CompleteAndMergeTracks(mapper_options, &mapper);
    num_changed_observations += FilterPoints(mapper_options, &mapper);
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < mapper_options.ba_global_max_refinement_change) {
      break;
    }
  }

  PrintHeading1("Extracting colors");
  reconstruction.ExtractColorsForAllImages(image_path);

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction);

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

namespace {

// Read the configuration of the camera rigs from a JSON file. The input images
// of a camera rig must be named consistently to assign them to the appropriate
// camera rig and the respective snapshots.
//
// An example configuration of a single camera rig:
// [
//   {
//     "ref_camera_id": 1,
//     "cameras":
//     [
//       {
//           "camera_id": 1,
//           "image_prefix": "left1_image"
//           "rel_tvec": [0, 0, 0],
//           "rel_qvec": [1, 0, 0, 0]
//       },
//       {
//           "camera_id": 2,
//           "image_prefix": "left2_image"
//           "rel_tvec": [0, 0, 0],
//           "rel_qvec": [0, 1, 0, 0]
//       },
//       {
//           "camera_id": 3,
//           "image_prefix": "right1_image"
//           "rel_tvec": [0, 0, 0],
//           "rel_qvec": [0, 0, 1, 0]
//       },
//       {
//           "camera_id": 4,
//           "image_prefix": "right2_image"
//           "rel_tvec": [0, 0, 0],
//           "rel_qvec": [0, 0, 0, 1]
//       }
//     ]
//   }
// ]
//
// The "camera_id" and "image_prefix" fields are required, whereas the
// "rel_tvec" and "rel_qvec" fields optionally specify the relative
// extrinsics of the camera rig in the form of a translation vector and a
// rotation quaternion. The relative extrinsics rel_qvec and rel_tvec transform
// coordinates from rig to camera coordinate space. If the relative extrinsics
// are not provided then they are automatically inferred from the
// reconstruction.
//
// This file specifies the configuration for a single camera rig and that you
// could potentially define multiple camera rigs. The rig is composed of 4
// cameras: all images of the first camera must have "left1_image" as a name
// prefix, e.g., "left1_image_frame000.png" or "left1_image/frame000.png".
// Images with the same suffix ("_frame000.png" and "/frame000.png") are
// assigned to the same snapshot, i.e., they are assumed to be captured at the
// same time. Only snapshots with the reference image registered will be added
// to the bundle adjustment problem. The remaining images will be added with
// independent poses to the bundle adjustment problem. The above configuration
// could have the following input image file structure:
//
//    /path/to/images/...
//        left1_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        left2_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        right1_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        right2_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//
std::vector<CameraRig> ReadCameraRigConfig(const std::string& rig_config_path,
                                           const Reconstruction& reconstruction,
                                           bool estimate_rig_relative_poses) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(rig_config_path.c_str(), pt);

  std::vector<CameraRig> camera_rigs;
  for (const auto& rig_config : pt) {
    CameraRig camera_rig;

    std::vector<std::string> image_prefixes;
    for (const auto& camera : rig_config.second.get_child("cameras")) {
      const int camera_id = camera.second.get<int>("camera_id");
      image_prefixes.push_back(camera.second.get<std::string>("image_prefix"));
      Eigen::Vector3d rel_tvec;
      Eigen::Vector4d rel_qvec;
      int index = 0;
      auto rel_tvec_node = camera.second.get_child_optional("rel_tvec");
      if (rel_tvec_node) {
        for (const auto& node : rel_tvec_node.get()) {
          rel_tvec[index++] = node.second.get_value<double>();
        }
      } else {
        estimate_rig_relative_poses = true;
      }
      index = 0;
      auto rel_qvec_node = camera.second.get_child_optional("rel_qvec");
      if (rel_qvec_node) {
        for (const auto& node : rel_qvec_node.get()) {
          rel_qvec[index++] = node.second.get_value<double>();
        }
      } else {
        estimate_rig_relative_poses = true;
      }

      camera_rig.AddCamera(camera_id, rel_qvec, rel_tvec);
    }

    camera_rig.SetRefCameraId(rig_config.second.get<int>("ref_camera_id"));

    std::unordered_map<std::string, std::vector<image_t>> snapshots;
    for (const auto image_id : reconstruction.RegImageIds()) {
      const auto& image = reconstruction.Image(image_id);
      for (const auto& image_prefix : image_prefixes) {
        if (StringContains(image.Name(), image_prefix)) {
          const std::string image_suffix =
              StringGetAfter(image.Name(), image_prefix);
          snapshots[image_suffix].push_back(image_id);
        }
      }
    }

    for (const auto& snapshot : snapshots) {
      bool has_ref_camera = false;
      for (const auto image_id : snapshot.second) {
        const auto& image = reconstruction.Image(image_id);
        if (image.CameraId() == camera_rig.RefCameraId()) {
          has_ref_camera = true;
        }
      }

      if (has_ref_camera) {
        camera_rig.AddSnapshot(snapshot.second);
      }
    }

    camera_rig.Check(reconstruction);
    if (estimate_rig_relative_poses) {
      PrintHeading2("Estimating relative rig poses");
      if (!camera_rig.ComputeRelativePoses(reconstruction)) {
        std::cout << "WARN: Failed to estimate rig poses from reconstruction; "
                     "cannot use rig BA"
                  << std::endl;
        return std::vector<CameraRig>();
      }
    }

    camera_rigs.push_back(camera_rig);
  }

  return camera_rigs;
}

}  // namespace

int RunRigBundleAdjuster(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string rig_config_path;
  bool estimate_rig_relative_poses = true;

  RigBundleAdjuster::Options rig_ba_options;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("rig_config_path", &rig_config_path);
  options.AddDefaultOption("estimate_rig_relative_poses",
                           &estimate_rig_relative_poses);
  options.AddDefaultOption("RigBundleAdjustment.refine_relative_poses",
                           &rig_ba_options.refine_relative_poses);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading1("Camera rig configuration");

  auto camera_rigs = ReadCameraRigConfig(rig_config_path, reconstruction,
                                         estimate_rig_relative_poses);

  BundleAdjustmentConfig config;
  for (size_t i = 0; i < camera_rigs.size(); ++i) {
    const auto& camera_rig = camera_rigs[i];
    PrintHeading2(StringPrintf("Camera Rig %d", i + 1));
    std::cout << StringPrintf("Cameras: %d", camera_rig.NumCameras())
              << std::endl;
    std::cout << StringPrintf("Snapshots: %d", camera_rig.NumSnapshots())
              << std::endl;

    // Add all registered images to the bundle adjustment configuration.
    for (const auto image_id : reconstruction.RegImageIds()) {
      config.AddImage(image_id);
    }
  }

  PrintHeading1("Rig bundle adjustment");

  BundleAdjustmentOptions ba_options = *options.bundle_adjustment;
  ba_options.solver_options.minimizer_progress_to_stdout = false;
  RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, config);
  CHECK(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

}  // namespace colmap
