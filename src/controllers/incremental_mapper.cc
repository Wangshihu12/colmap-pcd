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

#include "controllers/incremental_mapper.h"

#include "util/misc.h"

namespace colmap {
namespace {

size_t TriangulateImage(const IncrementalMapperOptions& options,
                        const Image& image, IncrementalMapper* mapper) {
  std::cout << "  => Continued observations: " << image.NumPoints3D()
            << std::endl;
  const size_t num_tris =
      mapper->TriangulateImage(options.Triangulation(), image.ImageId());
  std::cout << "  => Added observations: " << num_tris << std::endl;
  return num_tris;
}

/**
 * 全局束调整控制函数 - 根据不同条件选择合适的全局束调整方法
 * 
 * 该函数作为控制器，负责根据当前重建状态和用户设置，
 * 选择最合适的全局束调整方法：普通全局束调整、激光雷达约束的全局束调整或并行全局束调整。
 * 
 * @param options 增量式重建的控制选项集合
 * @param mapper 增量式重建器指针，执行实际重建过程
 */
void AdjustGlobalBundle(const IncrementalMapperOptions& options,
                        IncrementalMapper* mapper) {
  // 复制全局束调整选项，允许根据需要进行自定义修改
  BundleAdjustmentOptions custom_ba_options = options.GlobalBundleAdjustment();

  // 获取当前已注册图像的数量
  const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

  // 对于较小规模的重建（前几张图像），使用更严格的收敛标准
  // 这有助于建立稳定的初始结构，为后续重建打好基础
  const size_t kMinNumRegImagesForFastBA = 10; // 快速BA的最小图像数阈值
  if (num_reg_images < kMinNumRegImagesForFastBA) {
    // 增加优化精度：降低容差、增加迭代次数
    custom_ba_options.solver_options.function_tolerance /= 10;
    custom_ba_options.solver_options.gradient_tolerance /= 10;
    custom_ba_options.solver_options.parameter_tolerance /= 10;
    custom_ba_options.solver_options.max_num_iterations *= 2;
    custom_ba_options.solver_options.max_linear_solver_iterations = 200;
  }

  // 打印全局束调整开始的标题
  PrintHeading1("Global bundle adjustment");

  // 根据配置选择合适的全局束调整方法
  if (options.if_add_lidar_constraint) {
    // 如果启用激光雷达约束，使用基于激光雷达的全局束调整
    mapper->AdjustGlobalBundleByLidar(options.Mapper(), custom_ba_options);  
  } else {
    // 如果没有激光雷达约束，则在普通全局束调整和并行全局束调整之间选择
    // 满足以下条件时使用并行束调整：
    // 1. 用户启用了并行BA选项
    // 2. 不需要固定已存在的图像
    // 3. 已注册图像数量足够多
    // 4. 并行BA支持当前的优化配置和重建场景
    if (options.ba_global_use_pba && !options.fix_existing_images &&
        num_reg_images >= kMinNumRegImagesForFastBA &&
        ParallelBundleAdjuster::IsSupported(custom_ba_options,
                                            mapper->GetReconstruction())) {
      // 使用并行全局束调整，适合大规模场景
      mapper->AdjustParallelGlobalBundle(
          custom_ba_options, options.ParallelGlobalBundleAdjustment());
    } else {
      // 使用标准全局束调整
      mapper->AdjustGlobalBundle(options.Mapper(), custom_ba_options);
    }
  }
}

void IterativeLocalRefinement(const IncrementalMapperOptions& options,
                              const image_t image_id,
                              IncrementalMapper* mapper) {
  // mapper->ClearLidarPoints();
  auto ba_options = options.LocalBundleAdjustment();
  for (int i = 0; i < options.ba_local_max_refinements; ++i) {
    const auto report = mapper->AdjustLocalBundle(
        options.Mapper(), ba_options, options.Triangulation(), image_id,
        mapper->GetModifiedPoints3D());
    std::cout << "  => Merged observations: " << report.num_merged_observations
              << std::endl;
    std::cout << "  => Completed observations: "
              << report.num_completed_observations << std::endl;
    std::cout << "  => Filtered observations: "
              << report.num_filtered_observations << std::endl;
    const double changed =
        report.num_adjusted_observations == 0
            ? 0
            : (report.num_merged_observations +
               report.num_completed_observations +
               report.num_filtered_observations) /
                  static_cast<double>(report.num_adjusted_observations);
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < options.ba_local_max_refinement_change) {
      break;
    }
    // Only use robust cost function for first iteration.
    ba_options.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  }
  mapper->ClearModifiedPoints3D();

}

/**
 * 迭代式全局优化函数 - 循环优化重建结果直至收敛
 * 
 * 该函数执行迭代式的全局优化过程，包括重三角化、全局束调整、轨迹合并和过滤等步骤，
 * 通过多次迭代来提高三维重建结果的精度和完整性，直到达到收敛条件。
 * 
 * @param options 增量式重建的控制选项集合
 * @param mapper 增量式重建器指针，执行实际重建过程
 */
void IterativeGlobalRefinement(const IncrementalMapperOptions& options,
                               IncrementalMapper* mapper) {
  // 打印重三角化阶段标题
  PrintHeading1("Retriangulation");

  // 先执行一次轨迹完成和合并操作
  // 这将尝试为未三角化的特征点创建3D点，并合并重复的3D点轨迹
  CompleteAndMergeTracks(options, mapper);

  // 执行重三角化，尝试改进现有3D点的位置或重建失败的点
  // 并输出成功重三角化的观测数量
  std::cout << "  => Retriangulated observations: "
            << mapper->Retriangulate(options.Triangulation()) << std::endl;

  // 执行多轮迭代优化，直到达到最大迭代次数或收敛
  for (int i = 0; i < options.ba_global_max_refinements; ++i) {
    // 记录当前观测点总数，用于计算变化率
    const size_t num_observations =
        mapper->GetReconstruction().ComputeNumObservations();

    // 记录本轮迭代中变化的观测点数量
    size_t num_changed_observations = 0;

    // 执行全局束调整，优化所有相机参数和3D点坐标
    AdjustGlobalBundle(options, mapper);

    // 再次完成和合并轨迹，并累加变化的观测点数
    // 优化后的相机参数可能使之前失败的三角化成功
    num_changed_observations += CompleteAndMergeTracks(options, mapper);

    // 过滤低质量的3D点，并累加被过滤的观测点数
    num_changed_observations += FilterPoints(options, mapper);

    // 计算变化率：变化的观测点数占总观测点数的比例
    // 避免除以零的情况
    const double changed =
        num_observations == 0
            ? 0
            : static_cast<double>(num_changed_observations) / num_observations;

    // 输出本轮迭代的变化率
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;

    // 如果变化率低于阈值，认为已收敛，提前结束迭代
    if (changed < options.ba_global_max_refinement_change) {
      break;
    }
  }

  // 迭代结束后，过滤冗余或低质量的图像
  // 这可能移除注册失败或对重建贡献很少的图像
  FilterImages(options, mapper);
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   Reconstruction* reconstruction) {
  if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
    std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                              reconstruction->Image(image_id).Name().c_str(),
                              image_path.c_str())
              << std::endl;
  }
}

void WriteSnapshot(const Reconstruction& reconstruction,
                   const std::string& snapshot_path) {
  PrintHeading1("Creating snapshot");
  // Get the current timestamp in milliseconds.
  const size_t timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  // Write reconstruction to unique path with current timestamp.
  const std::string path =
      JoinPaths(snapshot_path, StringPrintf("%010d", timestamp));
  CreateDirIfNotExists(path);
  std::cout << "  => Writing to " << path << std::endl;
  reconstruction.Write(path);
}

}  // namespace

size_t FilterPoints(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper) {
  const size_t num_filtered_observations =
      mapper->FilterPoints(options.Mapper());
  std::cout << "  => Filtered observations: " << num_filtered_observations
            << std::endl;
  return num_filtered_observations;
}

size_t FilterImages(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper) {
  const size_t num_filtered_images = mapper->FilterImages(options.Mapper());
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
  return num_filtered_images;
}

size_t CompleteAndMergeTracks(const IncrementalMapperOptions& options,
                              IncrementalMapper* mapper) {
  const size_t num_completed_observations =
      mapper->CompleteTracks(options.Triangulation());
  std::cout << "  => Completed observations: " << num_completed_observations
            << std::endl;
  const size_t num_merged_observations =
      mapper->MergeTracks(options.Triangulation());
  std::cout << "  => Merged observations: " << num_merged_observations
            << std::endl;
  return num_completed_observations + num_merged_observations;
}

IncrementalMapper::Options IncrementalMapperOptions::Mapper() const {
  IncrementalMapper::Options options = mapper;
  options.first_image_fixed_frames = first_image_fixed_frames;
  options.min_proj_num = min_proj_num; 
  options.kdtree_max_search_range = kdtree_max_search_range;
  options.kdtree_min_search_range = kdtree_min_search_range;
  options.search_range_drop_speed = search_range_drop_speed;
  options.ba_spherical_search_radius = ba_spherical_search_radius;
  options.ba_match_features_threshold = ba_match_features_threshold;
  options.proj_max_dist_error = proj_max_dist_error;
  options.icp_max_dist_error = icp_max_dist_error;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  options.local_ba_num_images = ba_local_num_images;
  options.fix_existing_images = fix_existing_images;
  options.init_image_id1 = init_image_id1;
  options.init_image_id2 = init_image_id2;
  options.init_image_x = init_image_x;
  options.init_image_y = init_image_y;
  options.init_image_z = init_image_z;
  options.init_image_roll = init_image_roll;
  options.init_image_pitch = init_image_pitch;
  options.init_image_yaw = init_image_yaw;
  return options;
}

IncrementalTriangulator::Options IncrementalMapperOptions::Triangulation()
    const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

BundleAdjustmentOptions IncrementalMapperOptions::LocalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  // lidar related params
  options.if_add_lidar_constraint = if_add_lidar_constraint;
  options.lidar_pointcloud_path = lidar_pointcloud_path;
  options.proj_lidar_constraint_weight = proj_lidar_constraint_weight;
  options.icp_lidar_constraint_weight = icp_lidar_constraint_weight;
  options.icp_ground_lidar_constraint_weight = icp_ground_lidar_constraint_weight;

  options.if_add_lidar_corresponding = if_add_lidar_corresponding;
  
  options.ba_match_features_threshold = ba_match_features_threshold;

  options.solver_options.function_tolerance = ba_local_function_tolerance;
  options.solver_options.gradient_tolerance = 10.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_local_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = false;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  options.loss_function_scale = 1.0;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  return options;
}
lidar::PcdProjectionOptions IncrementalMapperOptions::PcdProjector() 
    const {
  lidar::PcdProjectionOptions options;
  options.depth_image_scale = depth_image_scale;
  options.choose_meter = static_cast<float>(choose_meter);
  options.max_proj_scale = max_proj_scale;
  options.min_proj_scale = min_proj_scale;
  options.min_proj_dist = min_proj_dist;
  options.min_lidar_proj_dist = min_lidar_proj_dist;
  options.if_save_depth_image = if_save_depth_image;
  options.original_image_folder = original_image_folder;
  options.depth_image_folder = depth_image_folder;
  options.if_save_lidar_frame = if_save_lidar_frame;
  options.lidar_frame_folder = lidar_frame_folder;
  options.submap_length = static_cast<float>(submap_length);//submap尺寸
  options.submap_width = static_cast<float>(submap_width);
  options.submap_height = static_cast<float>(submap_height);

  return options;
}

BundleAdjustmentOptions IncrementalMapperOptions::GlobalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.if_add_lidar_constraint = if_add_lidar_constraint;
  options.solver_options.function_tolerance = ba_global_function_tolerance;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = true;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  return options;
}


ParallelBundleAdjuster::Options
IncrementalMapperOptions::ParallelGlobalBundleAdjustment() const {
  ParallelBundleAdjuster::Options options;
  options.max_num_iterations = ba_global_max_num_iterations;
  options.print_summary = true;
  options.gpu_index = ba_global_pba_gpu_index;
  options.num_threads = num_threads;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  return options;
}

bool IncrementalMapperOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(max_num_models, 0);
  CHECK_OPTION_GT(max_model_overlap, 0);
  CHECK_OPTION_GE(min_model_size, 0);
  CHECK_OPTION_GT(init_num_trials, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GE(ba_local_num_images, 2);
  CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_images_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_images_freq, 0);
  CHECK_OPTION_GT(ba_global_points_freq, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_local_max_refinements, 0);
  CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GE(snapshot_images_freq, 0);
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Triangulation().Check());
  return true;
}

IncrementalMapperController::IncrementalMapperController(
    IncrementalMapperOptions* options, const std::string& image_path,
    const std::string& database_path, 
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(reconstruction_manager) {
  options ->original_image_folder = image_path; 
  CHECK(options_->Check());
  RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

/**
 * 增量式重建控制器的主运行函数 - 整个重建过程的入口点
 * 
 * 该函数是COLMAP增量式重建的顶层控制函数，负责初始化和执行完整的重建流程，
 * 包括数据库加载、初始重建以及在初始化失败时进行自适应参数放宽的尝试。
 */
void IncrementalMapperController::Run() {

  // 加载特征数据库，包含图像、特征点和匹配信息
  // 如果加载失败，直接终止重建过程
  if (!LoadDatabase()) {
    return;
  }

  // 如果启用了导入位姿先验选项，尝试加载已有的相机位姿
  // 这通常用于使用外部位姿信息辅助重建
  if (options_->if_import_pose_prior) {
    if (!LoadPose()) {
      return;
    }
  }

  // 创建初始重建选项的副本，用于后续可能的参数调整
  IncrementalMapper::Options init_mapper_options = options_->Mapper();

  // 使用初始参数尝试执行重建
  Reconstruct(init_mapper_options);

  // 定义最大放宽初始化参数的次数
  // 如果初始参数下重建失败，将逐步放宽约束条件再次尝试
  const size_t kNumInitRelaxations = 2;
  for (size_t i = 0; i < kNumInitRelaxations; ++i) {
    // 如果已经成功重建或用户中止，则退出放宽循环
    if (reconstruction_manager_->Size() > 0 || IsStopped()) {
      break;
    }

    // 第一次放宽：减少所需的内点数量
    // 这降低了图像对初始化时所需匹配点的数量门槛
    std::cout << "  => Relaxing the initialization constraints." << std::endl;
    init_mapper_options.init_min_num_inliers /= 2; // 将最小内点数减半
    Reconstruct(init_mapper_options); // 使用新参数再次尝试重建

    // 检查是否成功或中止
    if (reconstruction_manager_->Size() > 0 || IsStopped()) {
      break;
    }

    // 第二次放宽：减小三角化所需的最小角度
    // 这允许在视差较小的情况下也能初始化重建
    std::cout << "  => Relaxing the initialization constraints." << std::endl;
    init_mapper_options.init_min_tri_angle /= 2; // 将最小三角化角度减半
    Reconstruct(init_mapper_options); // 使用新参数再次尝试重建
  }

  // 打印整个重建过程的耗时（分钟为单位）
  GetTimer().PrintMinutes();
}

/**
 * 加载SfM数据库到内存缓存
 * 
 * 该方法负责从SQLite数据库文件中加载所有必要的SfM数据（图像、特征点、匹配等）
 * 到内存缓存中，以提高后续重建过程的访问速度
 * 
 * @return 数据库加载是否成功
 */
bool IncrementalMapperController::LoadDatabase() {

  //////////////////////////////////////////////////////////////////////////////
  // 处理图像名称过滤集合
  //////////////////////////////////////////////////////////////////////////////
  
  // 创建图像名称集合的副本，用于确定哪些图像需要被加载
  // 这个集合可能包含用户手动指定的图像名称
  std::unordered_set<std::string> image_names = options_->image_names;

  // 处理特殊情况：当存在一个重建并且用户手动指定了图像时
  // 需要确保已存在重建中的图像也被包含在加载范围内
  if (reconstruction_manager_->Size() == 1 && !options_->image_names.empty()) {
    // 获取已存在的重建对象引用
    const Reconstruction& reconstruction = reconstruction_manager_->Get(0);

    // 遍历该重建中所有已注册的图像
    for (const image_t image_id : reconstruction.RegImageIds()) {
      // 获取图像对象并提取图像名称
      const auto& image = reconstruction.Image(image_id);

      // 将已注册图像的名称添加到图像名称集合中
      // 这确保了即使用户手动指定图像列表，已存在重建中的图像也不会被忽略
      // 这对于增量重建的连续性很重要
      image_names.insert(image.Name());
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // 初始化数据库连接和计时器
  //////////////////////////////////////////////////////////////////////////////
  
  // 创建数据库连接对象，连接到指定路径的SQLite数据库文件
  Database database(database_path_);

  // 启动计时器，用于测量数据库加载的耗时
  Timer timer;
  timer.Start();

  //////////////////////////////////////////////////////////////////////////////
  // 加载数据库内容到内存缓存
  //////////////////////////////////////////////////////////////////////////////
  
  // 设置最小匹配数量阈值
  // 只有匹配数量达到此阈值的图像对才会被加载，用于过滤低质量的匹配
  const size_t min_num_matches = static_cast<size_t>(options_->min_num_matches);
  
  // 核心加载操作：将数据库内容加载到内存缓存
  // database_cache_: 数据库缓存对象，将数据库内容缓存在内存中，
  // 用于在重建多个模型时快速创建新的重建实例
  database_cache_.Load(database,                     // 数据库连接对象
                       min_num_matches,              // 最小匹配数量阈值
                       options_->ignore_watermarks,  // 是否忽略水印检测
                       image_names);                 // 要加载的图像名称集合
  std::cout << std::endl;
  timer.PrintMinutes();

  std::cout << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // 验证加载结果
  //////////////////////////////////////////////////////////////////////////////
  
  // 检查是否成功加载了图像数据
  if (database_cache_.NumImages() == 0) {
    std::cout << "WARNING: No images with matches found in the database."
              << std::endl
              << std::endl;
    return false;
  }

  return true;
}

int IncrementalMapperController::OriginImagesNum() {
  Database database(database_path_);
  int num = database.ReadAllImages().size();
  return num;
}

/**
 * 增量式三维重建主控函数 - 执行完整的SfM重建流程
 * 
 * 该函数是COLMAP增量式重建的核心控制器，管理整个重建过程，
 * 包括初始化、增量式图像注册、三角化、局部和全局优化等所有步骤。
 * 
 * @param init_mapper_options 增量式重建的初始配置选项
 */
void IncrementalMapperController::Reconstruct(
    const IncrementalMapper::Options& init_mapper_options) {
  // 定义重建失败时是否丢弃当前重建
  const bool kDiscardReconstruction = true;

  //////////////////////////////////////////////////////////////////////////////
  // Main loop
  //////////////////////////////////////////////////////////////////////////////

  // 创建增量式重建器对象
  IncrementalMapper mapper(&database_cache_);

  // 如果启用了导入位姿先验选项，加载已有的图像位姿
  if (options_->if_import_pose_prior) {
    mapper.LoadExistedImagePoses(image_poses_);
  }

  // 如果启用了激光雷达约束，加载点云数据
  if (options_->if_add_lidar_constraint || options_->if_add_lidar_corresponding){
    std::string path = options_->lidar_pointcloud_path;
    mapper.LoadPointcloud(path, options_->PcdProjector());
  }

  // 检查是否存在初始重建（用户导入的重建）
  const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
  // 确保最多只有一个初始重建
  CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                  "single reconstruction, but "
                                                  "multiple are given.";

  // 多次尝试重建循环，允许在初始化失败时尝试不同的初始图像对
  for (int num_trials = 0; num_trials < options_->init_num_trials;
       ++num_trials) {
    // 检查暂停和停止信号
    BlockIfPaused();
    if (IsStopped()) {
      break;
    }

    // 创建或使用重建对象
    size_t reconstruction_idx;
    if (!initial_reconstruction_given || num_trials > 0) {
      // 如果没有初始重建或者是重试，创建新的重建
      reconstruction_idx = reconstruction_manager_->Add();
    } else {
      // 使用已有的初始重建
      reconstruction_idx = 0;
    }
    Reconstruction& reconstruction =
        reconstruction_manager_->Get(reconstruction_idx);

    // 开始一个新的重建过程
    mapper.BeginReconstruction(&reconstruction);

    ////////////////////////////////////////////////////////////////////////////
    // Register initial pair
    ////////////////////////////////////////////////////////////////////////////

    // 如果当前重建中没有已注册的图像，需要选择并注册初始图像对
    if (reconstruction.NumRegImages() == 0) {
      // 初始化图像ID，可以是用户指定的或自动选择的
      image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
      image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

      // 如果没有指定初始图像对，自动寻找合适的初始对
      if (options_->init_image_id1 == -1 || options_->init_image_id2 == -1) {
        PrintHeading1("Finding good initial image pair");
        const bool find_init_success = mapper.FindInitialImagePair(
            init_mapper_options, &image_id1, &image_id2);
        // 如果找不到好的初始图像对，放弃当前尝试
        if (!find_init_success) {
          std::cout << "  => No good initial image pair found." << std::endl;
          mapper.EndReconstruction(kDiscardReconstruction);
          reconstruction_manager_->Delete(reconstruction_idx);
          break;
        }
      } else {
        // 检查用户指定的初始图像对是否存在
        if (!reconstruction.ExistsImage(image_id1) ||
            !reconstruction.ExistsImage(image_id2)) {
          std::cout << StringPrintf(
                           "  => Initial image pair #%d and #%d do not exist.",
                           image_id1, image_id2)
                    << std::endl;
          mapper.EndReconstruction(kDiscardReconstruction);
          reconstruction_manager_->Delete(reconstruction_idx);
          return;
        }
      }

      // 开始使用选定的初始图像对初始化重建
      PrintHeading1(StringPrintf("Initializing with image pair #%d and #%d",
                                 image_id1, image_id2));
      // input: IncrementalMapper::Options init_mapper_options
      // input: initial image pair

      // 根据是否使用激光雷达约束选择不同的初始化方法
      bool reg_init_success;
      if (options_->if_add_lidar_constraint){
        // 使用深度投影方法初始化
        reg_init_success = mapper.RegisterInitialImagePairByDepthProj(
          init_mapper_options, image_id1, image_id2);
      } else {
        // 使用传统方法初始化
        reg_init_success = mapper.RegisterInitialImagePair(
            init_mapper_options, image_id1, image_id2);
      } 
          
      // 初始化失败，提供可能的解决方案
      if (!reg_init_success) {
        std::cout << "  => Initialization failed - possible solutions:"
                  << std::endl
                  << "     - try to relax the initialization constraints"
                  << std::endl
                  << "     - manually select an initial image pair"
                  << std::endl;
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        break;
      }

      // 对初始结构执行全局束调整
      AdjustGlobalBundle(*options_, &mapper);

      // 过滤低质量的点和图像
      FilterPoints(*options_, &mapper);
      FilterImages(*options_, &mapper);
 
      // 如果初始化后没有成功注册图像或三角化点，放弃当前尝试
      if (reconstruction.NumRegImages() == 0 ||
          reconstruction.NumPoints3D() == 0) {
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        // 如果初始图像对是手动指定的，不再尝试其他初始对
        if (options_->init_image_id1 != -1 && options_->init_image_id2 != -1) {
          break;
        } else {
          continue;
        }
      }

      // 如果需要，提取并设置3D点的颜色
      if (options_->extract_colors) {
        ExtractColors(image_path_, image_id1, &reconstruction);
      }
    }
    
    // 初始图像对注册成功后，触发回调（通常用于可视化）
    Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

    ////////////////////////////////////////////////////////////////////////////
    // Incremental mapping
    ////////////////////////////////////////////////////////////////////////////

    // 记录重建状态，用于决定何时执行全局优化
    size_t snapshot_prev_num_reg_images = reconstruction.NumRegImages();
    size_t ba_prev_num_reg_images = reconstruction.NumRegImages();
    size_t ba_prev_num_points = reconstruction.NumPoints3D();

    // 增量式重建的主循环
    bool reg_next_success = true; // 注册下一张图像是否成功
    bool prev_reg_next_success = true; // 上一次注册是否成功
    while (reg_next_success) {
      // 检查暂停和停止信号
      BlockIfPaused();
      if (IsStopped()) {
        break;
      }

      reg_next_success = false;

      // 寻找下一批要注册的图像
      const std::vector<image_t> next_images =
          mapper.FindNextImages(options_->Mapper());

      // 如果没有更多图像可注册，结束增量式重建
      if (next_images.empty()) {
        break;
      }

      // 尝试注册候选图像列表中的图像
      for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
        const image_t next_image_id = next_images[reg_trial];
        const Image& next_image = reconstruction.Image(next_image_id);

        PrintHeading1(StringPrintf("Registering image #%d (%d)", next_image_id,
                                   reconstruction.NumRegImages() + 1));

        // 显示该图像能看到多少已重建的3D点
        std::cout << StringPrintf("  => Image sees %d / %d points",
                                  next_image.NumVisiblePoints3D(),
                                  next_image.NumObservations())
                  << std::endl;

        // 注册下一张图像
        reg_next_success =
            mapper.RegisterNextImage(options_->Mapper(), next_image_id);

        // 如果注册成功
        if (reg_next_success) {
          // 清除前一次的激光雷达点匹配结果
          mapper.ClearLidarPoints();
          // 对新注册图像执行三角化，创建新的3D点
          TriangulateImage(*options_, next_image, &mapper);
          // 对新注册图像执行局部优化
          IterativeLocalRefinement(*options_, next_image_id, &mapper);

          // 检查是否需要执行全局优化
          // 条件：已注册图像数量或3D点数量相对上次全局BA有显著增加
          if (reconstruction.NumRegImages() >=
                  options_->ba_global_images_ratio * ba_prev_num_reg_images ||
              reconstruction.NumRegImages() >=
                  options_->ba_global_images_freq + ba_prev_num_reg_images ||
              reconstruction.NumPoints3D() >=
                  options_->ba_global_points_ratio * ba_prev_num_points ||
              reconstruction.NumPoints3D() >=
                  options_->ba_global_points_freq + ba_prev_num_points) {

            // 执行全局优化
            IterativeGlobalRefinement(*options_, &mapper);

            // 更新上次全局优化时的状态
            ba_prev_num_points = reconstruction.NumPoints3D();
            ba_prev_num_reg_images = reconstruction.NumRegImages();
          }

          // 如果需要，提取并设置新注册图像中3D点的颜色
          if (options_->extract_colors) {
            ExtractColors(image_path_, next_image_id, &reconstruction);
          }

          // 如果达到快照频率，保存当前重建的快照
          if (options_->snapshot_images_freq > 0 &&
              reconstruction.NumRegImages() >=
                  options_->snapshot_images_freq +
                      snapshot_prev_num_reg_images) {
            snapshot_prev_num_reg_images = reconstruction.NumRegImages();
            WriteSnapshot(reconstruction, options_->snapshot_path);
          }

          // 触发图像注册成功的回调
          Callback(NEXT_IMAGE_REG_CALLBACK);

          // 成功注册一张图像后，退出当前尝试循环
          break;
        } else {
          // 注册失败，尝试下一张候选图像
          std::cout << "  => Could not register, trying another image."
                    << std::endl;

          // 如果初始阶段长时间无法继续注册新图像，考虑放弃当前初始对
          const size_t kMinNumInitialRegTrials = 30;
          if (reg_trial >= kMinNumInitialRegTrials &&
              reconstruction.NumRegImages() <
                  static_cast<size_t>(options_->min_model_size)) {
            break;
          }
        }
      }

      // 检查当前模型与之前模型的重叠图像数量
      // 如果重叠过多，停止当前模型的重建（用于多模型重建场景）
      const size_t max_model_overlap =
          static_cast<size_t>(options_->max_model_overlap);
      if (mapper.NumSharedRegImages() >= max_model_overlap) {
        break;
      }

      // 如果本轮没有成功注册任何图像但上一轮成功了
      // 尝试执行一次全局优化后再试一次注册
      // 这是最后的"挽救"措施
      if (!reg_next_success && prev_reg_next_success) {
        reg_next_success = true; // 允许再尝试一次
        prev_reg_next_success = false;
        IterativeGlobalRefinement(*options_, &mapper);
      } else {
        prev_reg_next_success = reg_next_success;
      }
    }

    // 如果收到停止信号，结束当前重建但不丢弃结果
    if (IsStopped()) {
      const bool kDiscardReconstruction = false;
      mapper.EndReconstruction(kDiscardReconstruction);
      break;
    }

    // 如果最后一次增量BA不是全局性的，执行最终的全局优化
    if (reconstruction.NumRegImages() >= 2 &&
        reconstruction.NumRegImages() != ba_prev_num_reg_images &&
        reconstruction.NumPoints3D() != ba_prev_num_points) {
      IterativeGlobalRefinement(*options_, &mapper);
    }

    // 根据重建的规模决定是否保留当前重建
    // 对于小型图像集合，放宽最小模型大小的要求
    const size_t min_model_size =
        std::min(database_cache_.NumImages(),
                 static_cast<size_t>(options_->min_model_size));

    // 如果启用了多模型重建且当前模型太小，或者没有注册任何图像，丢弃当前重建
    if ((options_->multiple_models &&
         reconstruction.NumRegImages() < min_model_size) ||
        reconstruction.NumRegImages() == 0) {
      mapper.EndReconstruction(kDiscardReconstruction);
      reconstruction_manager_->Delete(reconstruction_idx);
    } else {
      // 否则保留当前重建
      const bool kDiscardReconstruction = false;
      mapper.EndReconstruction(kDiscardReconstruction);
    }

    // 触发最后一张图像注册的回调
    Callback(LAST_IMAGE_REG_CALLBACK);

    // 决定是否继续尝试创建更多模型
    // 如果满足以下任一条件，停止创建新模型：
    // 1. 已经有初始重建
    // 2. 不允许多模型重建
    // 3. 已达到最大模型数量
    // 4. 几乎所有图像都已注册
    const size_t max_num_models = static_cast<size_t>(options_->max_num_models);
    if (initial_reconstruction_given || !options_->multiple_models ||
        reconstruction_manager_->Size() >= max_num_models ||
        mapper.NumTotalRegImages() >= database_cache_.NumImages() - 1) {
      break;
    }

  }
}

bool IncrementalMapperController::LoadPose() {

  std::ifstream read_pose;
  read_pose.open(options_->image_pose_prior_path, std::ios::in);
  if (read_pose.is_open()){
    std::string str;
    bool end_header_show = false;
    image_t image_id = 0;
    while (getline(read_pose, str)){
      if (end_header_show){
        image_id += 1;
        bool exist_nan = false;
        std::stringstream ss(str);
        std::stringstream ifnan(str);
        std::string s;
        while (ifnan >> s) {
          if (s == "nan") {
            exist_nan = true;
            break;
          }
        }
        if (exist_nan) {
          continue;
        }
        double d;
        std::vector<double> pose;
        while (ss >> d){
          pose.push_back(d);

        }

        double t_x = -static_cast<double>(pose[1]);
        double t_y = -static_cast<double>(pose[2]);
        double t_z = static_cast<double>(pose[0]);
        double roll = static_cast<double>(pose[3]);
        double pitch = -static_cast<double>(pose[4]);
        double yaw = -static_cast<double>(pose[5]);

        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitZ()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitY()));
      
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix = yawAngle * pitchAngle * rollAngle;
        Eigen::Matrix3d R_wc = rotation_matrix;

        Eigen::Vector3d t_wc;
        t_wc << t_x, t_y, t_z;

        Eigen::Matrix3d R_cw = R_wc.transpose();
        Eigen::Vector3d t_cw = - R_cw * t_wc;

        Eigen::Quaterniond q_cw(R_cw);
        std::vector<double> trans_pose {t_cw(0), t_cw(1), t_cw(2), 
                              q_cw.w(), q_cw.x(), q_cw.y(), q_cw.z()};
        image_poses_.emplace(std::make_pair(image_id,trans_pose));

      } else {
        if (str == "end_header"){
          end_header_show = true;
        }
      }
    }
  } else{
    std::cout << "Please note，failed to open the pose file!" << std::endl;
    return false;
  }

  read_pose.close();
  std::cout<<"Read " << image_poses_.size() << " poses from"<<std::endl
  << options_->image_pose_prior_path << std::endl
  << std::endl;
  return true;

}


}  // namespace colmap
