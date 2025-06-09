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

#include "sfm/incremental_mapper.h"

#include <array>
#include <fstream>

#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/pose.h"
#include "util/bitmap.h"
#include "util/misc.h"

namespace colmap {
namespace {

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
  std::sort(image_ranks.begin(), image_ranks.end(),
            [](const std::pair<image_t, float>& image1,
               const std::pair<image_t, float>& image2) {
              return image1.second > image2.second;
            });

  sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
  for (const auto& image : image_ranks) {
    sorted_images_ids->push_back(image.first);
  }

  image_ranks.clear();
}

float RankNextImageMaxVisiblePointsNum(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D());
}

float RankNextImageMaxVisiblePointsRatio(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D()) /
         static_cast<float>(image.NumObservations());
}

float RankNextImageMinUncertainty(const Image& image) {
  return static_cast<float>(image.Point3DVisibilityScore());
}

}  // namespace

bool IncrementalMapper::Options::Check() const {
  CHECK_OPTION_GT(init_min_num_inliers, 0);
  CHECK_OPTION_GT(init_max_error, 0.0);
  CHECK_OPTION_GE(init_max_forward_motion, 0.0);
  CHECK_OPTION_LE(init_max_forward_motion, 1.0);
  CHECK_OPTION_GE(init_min_tri_angle, 0.0);
  CHECK_OPTION_GE(init_max_reg_trials, 1);
  CHECK_OPTION_GT(abs_pose_max_error, 0.0);
  CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
  CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
  CHECK_OPTION_GE(local_ba_num_images, 2);
  CHECK_OPTION_GE(local_ba_min_tri_angle, 0.0);
  CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GE(max_extra_param, 0.0);
  CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
  CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
  CHECK_OPTION_GE(max_reg_trials, 1);
  return true;
}

IncrementalMapper::IncrementalMapper(const DatabaseCache* database_cache)
    : database_cache_(database_cache),
      reconstruction_(nullptr),
      triangulator_(nullptr),
      num_total_reg_images_(0),
      num_shared_reg_images_(0),
      prev_init_image_pair_id_(kInvalidImagePairId) {}

void IncrementalMapper::LoadExistedImagePoses(std::map<uint32_t, std::vector<double>>& poses){
  existed_poses_ = poses;
  if_import_pose_prior_ = true;
}

/**
 * 开始重建过程 - 初始化增量式SfM重建器
 * 
 * 该方法是增量式重建的入口点，负责设置所有必要的数据结构和状态，
 * 为后续的图像注册、三角化和束调整做准备
 * 
 * @param reconstruction 重建对象指针，用于存储和管理3D重建数据
 */
void IncrementalMapper::BeginReconstruction(Reconstruction* reconstruction) {
  // 确保当前没有正在进行的重建过程
  // 这是一个安全检查，防止重复初始化或状态混乱
  CHECK(reconstruction_ == nullptr);

  // 设置重建对象指针，这是整个重建过程的核心数据结构
  // 它将存储相机参数、图像位姿、3D点等所有重建信息
  reconstruction_ = reconstruction;

  // 从数据库缓存中加载重建数据
  // 包括相机内参、图像信息、特征点等基础数据
  reconstruction_->Load(*database_cache_);

  // 设置对应关系图，建立图像间特征点的对应关系
  // 这是增量式重建中查找图像间匹配关系的关键数据结构
  reconstruction_->SetUp(&database_cache_->CorrespondenceGraph());

  // 创建增量式三角化器
  // 负责在重建过程中逐步添加新的3D点，是增量式SfM的核心组件之一
  triangulator_ = std::make_unique<IncrementalTriangulator>(
      &database_cache_->CorrespondenceGraph(), reconstruction);

  // 重置共享注册图像计数器
  // 用于统计在多个重建中被注册的图像数量
  num_shared_reg_images_ = 0;

  // 清空每个相机的已注册图像计数表
  // 这个映射表记录每个相机模型对应的已注册图像数量
  // 在束调整时用于决定是否固定相机参数
  num_reg_images_per_camera_.clear();

  // 为已经注册的图像触发注册事件
  // 如果重建对象中已有注册的图像，需要更新相关统计信息
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    RegisterImageEvent(image_id);
  }

  // 记录已存在的图像ID集合
  // 这用于区分新注册的图像和预先存在的图像
  // 在某些配置下，预先存在的图像位姿可能需要保持固定
  existing_image_ids_ =
      std::unordered_set<image_t>(reconstruction->RegImageIds().begin(),
                                  reconstruction->RegImageIds().end());

  // 重置上一次初始化使用的图像对ID
  // 用于避免重复计算相同图像对的两视图几何关系
  prev_init_image_pair_id_ = kInvalidImagePairId;

  // 重置上一次计算的两视图几何关系
  // 存储最近一次成功计算的图像对几何关系，用于优化计算效率
  prev_init_two_view_geometry_ = TwoViewGeometry();

  // 清空已过滤图像列表
  // 记录因质量问题被过滤掉的图像，避免重复尝试注册
  filtered_images_.clear();

  // 清空图像注册尝试次数记录
  // 记录每个图像尝试注册的次数，防止无限重试
  num_reg_trials_.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
  CHECK_NOTNULL(reconstruction_);

  if (discard) {
    for (const image_t image_id : reconstruction_->RegImageIds()) {
      DeRegisterImageEvent(image_id);
    }
  }

  reconstruction_->TearDown();
  reconstruction_ = nullptr;
  triangulator_.reset();
}
void IncrementalMapper::LoadPointcloud(std::string& pointcloud_path, 
                                       const lidar::PcdProjectionOptions& pp_options){
  if (pointcloud_path == ""){
    std::cout << "Pose file path undefined." << std::endl;
    std::cout << std::endl;
  }                                    
  lidar_pointcloud_process_.reset(new lidar::PointCloudProcess(pointcloud_path));
  if (!lidar_pointcloud_process_->Initialize(pp_options)){
    std::cout<< "Error reading point cloud." << std::endl;
    std::cout << std::endl;

  }
}

// 查找合适的初始图像对用于增量式三维重建
// 输入参数：
//   - options: 算法配置选项
//   - image_id1, image_id2: 输出参数，存储找到的初始图像对的ID
// 返回值：
//   - true: 成功找到合适的初始图像对
//   - false: 未能找到合适的初始图像对
bool IncrementalMapper::FindInitialImagePair(const Options& options,
                                             image_t* image_id1,
                                             image_t* image_id2) {
  CHECK(options.Check());

  // 用于存储候选的第一张图像ID列表
  std::vector<image_t> image_ids1;

  // 处理三种情况：1.只提供了第一张图像 2.只提供了第二张图像 3.没有提供任何图像
  if (*image_id1 != kInvalidImageId && *image_id2 == kInvalidImageId) {
    // 情况1：只提供了第一张图像ID
    if (!database_cache_->ExistsImage(*image_id1)) {
      // 如果指定的图像不存在，返回失败
      return false;
    }
    // 将提供的图像ID添加为候选
    image_ids1.push_back(*image_id1);
  } else if (*image_id1 == kInvalidImageId && *image_id2 != kInvalidImageId) {
    // 情况2：只提供了第二张图像ID
    if (!database_cache_->ExistsImage(*image_id2)) {
      // 如果指定的图像不存在，返回失败
      return false;
    }
    image_ids1.push_back(*image_id2);
  } else {
    // 情况3：没有提供有效的初始图像
    // 调用函数自动选择第一张图像（通常基于特征点数量或连接性）
    image_ids1 = FindFirstInitialImage(options);
  }

  // 遍历所有候选的第一张图像
  for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
    // 设置当前尝试的第一张图像ID
    *image_id1 = image_ids1[i1];

    // 根据第一张图像查找合适的第二张图像（通常基于共视特征点数量）
    const std::vector<image_t> image_ids2 =
        FindSecondInitialImage(options, *image_id1);

    // 遍历所有候选的第二张图像
    for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
      // 设置当前尝试的第二张图像ID
      *image_id2 = image_ids2[i2];

      // 计算图像对的唯一标识符
      const image_pair_t pair_id =
          Database::ImagePairToPairId(*image_id1, *image_id2);

      // 如果这个图像对已经尝试过，则跳过
      // 这避免了重复计算和无限循环
      if (init_image_pairs_.count(pair_id) > 0) {
        continue;
      }

      // 将当前图像对标记为已尝试
      init_image_pairs_.insert(pair_id);

      // 尝试为这对图像估计初始的两视图几何关系（相对位姿）
      // 这通常包括估计基础矩阵或本质矩阵，以及三角化初始点云
      if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
        // 如果成功估计两视图几何，返回成功
        return true;
      }
    }
  }

  // 如果遍历完所有可能的图像对仍未找到合适的初始对
    // 设置图像ID为无效值并返回失败
  *image_id1 = kInvalidImageId;
  *image_id2 = kInvalidImageId;

  return false;
}

std::vector<image_t> IncrementalMapper::FindNextImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  std::function<float(const Image&)> rank_image_func;
  switch (options.image_selection_method) {
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
      rank_image_func = RankNextImageMaxVisiblePointsNum;
      break;
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
      rank_image_func = RankNextImageMaxVisiblePointsRatio;
      break;
    case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
      rank_image_func = RankNextImageMinUncertainty;
      break;
  }

  std::vector<std::pair<image_t, float>> image_ranks;
  std::vector<std::pair<image_t, float>> other_image_ranks;

  // Append images that have not failed to register before.
  for (const auto& image : reconstruction_->Images()) {
    // Skip images that are already registered.
    if (image.second.IsRegistered()) {
      continue;
    }

    // Only consider images with a sufficient number of visible points.
    if (image.second.NumVisiblePoints3D() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
      continue;
    }

    // Only try registration for a certain maximum number of times.
    const size_t num_reg_trials = num_reg_trials_[image.first];
    if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
      continue;
    }

    // If image has been filtered or failed to register, place it in the
    // second bucket and prefer images that have not been tried before.
    const float rank = rank_image_func(image.second);
    if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
      image_ranks.emplace_back(image.first, rank);
    } else {
      other_image_ranks.emplace_back(image.first, rank);
    }
  }

  std::vector<image_t> ranked_images_ids;
  SortAndAppendNextImages(image_ranks, &ranked_images_ids);
  SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

  return ranked_images_ids;
}

bool IncrementalMapper::RegisterInitialImagePair(const Options& options,
                                                 const image_t image_id1,
                                                 const image_t image_id2) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_EQ(reconstruction_->NumRegImages(), 0);

  CHECK(options.Check());

  init_num_reg_trials_[image_id1] += 1;
  init_num_reg_trials_[image_id2] += 1;
  num_reg_trials_[image_id1] += 1;
  num_reg_trials_[image_id2] += 1;

  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  init_image_pairs_.insert(pair_id);

  Image& image1 = reconstruction_->Image(image_id1);
  const Camera& camera1 = reconstruction_->Camera(image1.CameraId());

  Image& image2 = reconstruction_->Image(image_id2);
  const Camera& camera2 = reconstruction_->Camera(image2.CameraId());

  //////////////////////////////////////////////////////////////////////////////
  // Estimate two-view geometry
  //////////////////////////////////////////////////////////////////////////////

  if (!EstimateInitialTwoViewGeometry(options, image_id1, image_id2)) {
    return false;
  }
  image1.Qvec() = ComposeIdentityQuaternion();
  image1.Tvec() = Eigen::Vector3d(0, 0, 0);

  image2.Qvec() = prev_init_two_view_geometry_.qvec;
  image2.Tvec() = prev_init_two_view_geometry_.tvec;

  const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();
  const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();
  const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
  const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();

  //////////////////////////////////////////////////////////////////////////////
  // Update Reconstruction
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id1);
  reconstruction_->RegisterImage(image_id2);
  RegisterImageEvent(image_id1);
  RegisterImageEvent(image_id2);

  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  const FeatureMatches& corrs =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);

  // Add 3D point tracks.
  Track track;
  track.Reserve(2);
  track.AddElement(TrackElement());
  track.AddElement(TrackElement());
  track.Element(0).image_id = image_id1;
  track.Element(1).image_id = image_id2;
  for (const auto& corr : corrs) {
    const Eigen::Vector2d point1_N =
        camera1.ImageToWorld(image1.Point2D(corr.point2D_idx1).XY());
    const Eigen::Vector2d point2_N =
        camera2.ImageToWorld(image2.Point2D(corr.point2D_idx2).XY());
    const Eigen::Vector3d& xyz =
        TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);
    const double tri_angle =
        CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
    if (tri_angle >= min_tri_angle_rad &&
        HasPointPositiveDepth(proj_matrix1, xyz) &&
        HasPointPositiveDepth(proj_matrix2, xyz)) {
      track.Element(0).point2D_idx = corr.point2D_idx1;
      track.Element(1).point2D_idx = corr.point2D_idx2;
      reconstruction_->AddPoint3D(xyz, track);
    }
  }

  return true;
}

/**
 * 基于深度投影方法注册初始图像对
 * 
 * 该函数利用激光雷达点云数据辅助初始化第一对图像，与传统方法不同，
 * 它先为第一张图像赋予初始位姿，然后使用点云数据获取特征点的3D坐标，
 * 最后通过PnP算法估计第二张图像的位姿，建立初始重建结构。
 * 
 * @param options 重建配置选项
 * @param image_id1 第一张图像ID
 * @param image_id2 第二张图像ID
 * @return 初始化是否成功
 */
bool IncrementalMapper::RegisterInitialImagePairByDepthProj(const Options& options,
                                                            const image_t image_id1,
                                                            const image_t image_id2){

  // 确保重建对象存在且尚未注册任何图像
  CHECK_NOTNULL(reconstruction_);
  CHECK_EQ(reconstruction_->NumRegImages(), 0);

  // 检查选项参数是否合法
  CHECK(options.Check());

  // 更新初始化尝试计数器和总注册尝试计数器
  init_num_reg_trials_[image_id1] += 1;
  init_num_reg_trials_[image_id2] += 1;
  num_reg_trials_[image_id1] += 1;
  num_reg_trials_[image_id2] += 1;

  // 计算图像对的唯一ID并记录为已尝试
  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  init_image_pairs_.insert(pair_id);

  // 获取图像和相机对象的引用
  Image& image1 = reconstruction_->Image(image_id1);
  Camera& camera1 = reconstruction_->Camera(image1.CameraId());

  Image& image2 = reconstruction_->Image(image_id2);
  Camera& camera2 = reconstruction_->Camera(image2.CameraId());

  // 为第一张图像指定初始位姿
  // 根据配置选项中的欧拉角(roll, pitch, yaw)和位置坐标创建旋转矩阵和平移向量
  double roll = DegToRad(options.init_image_roll);
  double pitch = -DegToRad(options.init_image_pitch);
  double yaw = -DegToRad(options.init_image_yaw);
  //Eigen::Vector3d eulerAngle(roll, pitch, yaw);

  // 使用轴角表示创建旋转矩阵
  Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitZ()));
  Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitX()));
  Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitY()));
 
  // 组合欧拉角为完整的旋转矩阵
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix = yawAngle * pitchAngle * rollAngle;

  // 设置初始平移向量，注意坐标系转换
  Eigen::Vector3d t_init(-options.init_image_y, -options.init_image_z, options.init_image_x);

  // 世界坐标系到相机坐标系的变换
  Eigen::Matrix3d R_wc = rotation_matrix; // 世界到相机的旋转
  Eigen::Vector3d t_wc = t_init; // 世界到相机的平移

  // 相机坐标系到世界坐标系的变换（COLMAP内部使用）
  Eigen::Matrix3d R_cw = R_wc.transpose(); // 相机到世界的旋转
  Eigen::Vector3d t_cw = - R_cw * t_wc; // 相机到世界的平移

  // 将旋转矩阵转换为四元数
  Eigen::Quaterniond q_cw(R_cw);
  Eigen::Vector4d q_cw_v;
  q_cw_v << q_cw.w(),q_cw.x(),q_cw.y(),q_cw.z(); // COLMAP使用w,x,y,z顺序

  // 设置第一张图像的位姿
  image1.Qvec() = q_cw_v;
  image1.Tvec() = t_cw;

  // 如果启用了位姿先验导入，尝试使用已有的位姿数据
  if (if_import_pose_prior_) {
    // 查找第一张图像的先验位姿
    auto iter = existed_poses_.find(image1.ImageId());
    if (iter != existed_poses_.end()){
      std::vector<double> pose1 = iter->second;
      Eigen::Vector4d q_cw1;
      Eigen::Vector3d t_cw1;
      t_cw1 << pose1[0], pose1[1], pose1[2];
      q_cw1 << pose1[3], pose1[4], pose1[5], pose1[6];
      image1.SetQvec(q_cw1);
      image1.SetTvec(t_cw1);
    }

    // 查找第二张图像的先验位姿
    iter = existed_poses_.find(image2.ImageId());
    if (iter != existed_poses_.end()){
      std::vector<double> pose2 = iter->second;
      Eigen::Vector4d q_cw2;
      Eigen::Vector3d t_cw2;
      t_cw2 << pose2[0], pose2[1], pose2[2];
      q_cw2 << pose2[3], pose2[4], pose2[5], pose2[6];
      image2.SetQvec(q_cw2);
      image2.SetTvec(t_cw2);
    }

  }
  // image1.Qvec() = ComposeIdentityQuaternion();
  // image1.Tvec() = Eigen::Vector3d(0, 0, 0);

  // 获取两图像间的特征匹配
  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  const FeatureMatches matches =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  // 准备数据结构存储第一张图像的2D点和对应的3D点
  std::vector<std::pair<Eigen::Vector2d, bool>,Eigen::aligned_allocator<std::pair<Eigen::Vector2d, bool>>> image1_point2ds;
  std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> image2_point2ds;
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> image1_pt_xyzs;
  image1_point2ds.reserve(matches.size());
  image1_pt_xyzs.reserve(matches.size());

  // 收集匹配点的2D坐标
  for (const auto match : matches){
    const Point2D point2D_1 = image1.Point2D(match.point2D_idx1);
    image1_point2ds.push_back({point2D_1.XY(),false});
    const Point2D point2D_2 = image2.Point2D(match.point2D_idx2);
    image2_point2ds.push_back(point2D_2.XY());
  }
  
  // 使用激光雷达点云投影获取第一张图像特征点的3D坐标
  lidar_pointcloud_process_->pcd_proj_->SetNewImage(image1,camera1,image1_point2ds,image1_pt_xyzs);

  // 准备PnP所需的数据：成功获取3D坐标的点
  std::vector<Eigen::Vector2d> tri_points2D;  // 第二张图像中的2D点
  std::vector<Eigen::Vector3d> tri_points3D;  // 对应的3D点
  std::vector<point2D_t> image1_idxs; // 第一张图像中的点索引
  std::vector<point2D_t> image2_idxs; // 第二张图像中的点索引

  // 筛选成功获取3D坐标的点对
  for (int i = 0; i < image1_point2ds.size(); i++){
    if (image1_point2ds[i].second == false) continue; // 跳过未获取3D坐标的点

    tri_points2D.push_back(image2_point2ds[i]);
    tri_points3D.push_back(image1_pt_xyzs[i]);
    image1_idxs.push_back(matches[i].point2D_idx1);
    image2_idxs.push_back(matches[i].point2D_idx2);
  }

  // 配置绝对位姿估计选项
  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.num_threads = options.num_threads;
  abs_pose_options.num_focal_length_samples = 30;
  abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
  abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;
  // 使用高置信度避免P3P RANSAC过早终止
  abs_pose_options.ransac_options.min_num_trials = 100;
  abs_pose_options.ransac_options.max_num_trials = 10000;
  abs_pose_options.ransac_options.confidence = 0.99999;
  abs_pose_options.estimate_focal_length = false; // 不估计焦距

  // 配置位姿优化选项
  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  abs_pose_refinement_options.refine_focal_length = false;
  abs_pose_refinement_options.refine_extra_params = false;

  // 使用PnP算法估计第二张图像的位姿
  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
                            &image2.Qvec(), &image2.Tvec(), &camera2, &num_inliers,
                            &inlier_mask)) {
    return false; // 位姿估计失败
  }

  // 检查内点数量是否满足要求
  if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  // 优化第二张图像的位姿
  if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
                          tri_points2D, tri_points3D, &image2.Qvec(),
                          &image2.Tvec(), &camera2)) {
    return false;
  }

  // 注册两张图像到重建中
  reconstruction_->RegisterImage(image_id1);
  reconstruction_->RegisterImage(image_id2);

  // 触发图像注册事件
  RegisterImageEvent(image_id1);
  RegisterImageEvent(image_id2);

  // 创建特征点轨迹，每个轨迹连接两张图像中的对应点
  Track track;
  track.Reserve(2);
  track.AddElement(TrackElement());
  track.AddElement(TrackElement());
  track.Element(0).image_id = image_id1;
  track.Element(1).image_id = image_id2;

  // 为内点创建3D点和轨迹
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) { // 只处理内点
      track.Element(0).point2D_idx = image1_idxs[i];
      track.Element(1).point2D_idx = image2_idxs[i];
      const Eigen::Vector3d xyz = tri_points3D[i]; // 使用3D点坐标
      reconstruction_->AddPoint3D(xyz, track); // 添加到重建中
    }
  }

  return true;
}

bool IncrementalMapper::RegisterNextImage(const Options& options,
                                          const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_GE(reconstruction_->NumRegImages(), 2);

  CHECK(options.Check());

  Image& image = reconstruction_->Image(image_id);
  Camera& camera = reconstruction_->Camera(image.CameraId());

  CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

  num_reg_trials_[image_id] += 1;

  // Check if enough 2D-3D correspondences.
  if (image.NumVisiblePoints3D() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

    if (if_import_pose_prior_) {
      auto iter = existed_poses_.find(image_id);
      if (iter != existed_poses_.end()){
        std::vector<double> pose = iter -> second;
        Eigen::Vector4d q_cw;
        Eigen::Vector3d t_cw;
        t_cw << pose[0], pose[1], pose[2];
        q_cw << pose[3], pose[4], pose[5], pose[6];
        image.SetQvec(q_cw);
        image.SetTvec(t_cw);
      }

  }

  //////////////////////////////////////////////////////////////////////////////
  // Search for 2D-3D correspondences
  //////////////////////////////////////////////////////////////////////////////

  
  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();

  std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
  std::vector<Eigen::Vector2d> tri_points2D;
  std::vector<Eigen::Vector3d> tri_points3D;

  std::unordered_set<point3D_t> corr_point3D_ids;
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);

    corr_point3D_ids.clear();
    for (const auto& corr :
         correspondence_graph.FindCorrespondences(image_id, point2D_idx)) {
      const Image& corr_image = reconstruction_->Image(corr.image_id);
      // If this image hasn't been registered, ignore this image
      if (!corr_image.IsRegistered()) {
        continue;
      }
      const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }

      // Avoid duplicate correspondences.
      if (corr_point3D_ids.count(corr_point2D.Point3DId()) > 0) {
        continue;
      }
      const Camera& corr_camera =
          reconstruction_->Camera(corr_image.CameraId());

      // Avoid correspondences to images with bogus camera parameters.
      if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                     options.max_focal_length_ratio,
                                     options.max_extra_param)) {
        continue;
      }

      const Point3D& point3D =
          reconstruction_->Point3D(corr_point2D.Point3DId());

      tri_corrs.emplace_back(point2D_idx, corr_point2D.Point3DId());
      corr_point3D_ids.insert(corr_point2D.Point3DId());
      tri_points2D.push_back(point2D.XY());
      tri_points3D.push_back(point3D.XYZ());
    }
  }

  // The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
  // can only differ, when there are images with bogus camera parameters, and
  // hence we skip some of the 2D-3D correspondences.
  if (tri_points2D.size() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 2D-3D estimation
  //////////////////////////////////////////////////////////////////////////////

  // Only refine / estimate focal length, if no focal length was specified
  // (manually or through EXIF) and if it was not already estimated previously
  // from another image (when multiple images share the same camera
  // parameters)

  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.num_threads = options.num_threads;
  abs_pose_options.num_focal_length_samples = 30;
  abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
  abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;
  // Use high confidence to avoid preemptive termination of P3P RANSAC
  // - too early termination may lead to bad registration.
  abs_pose_options.ransac_options.min_num_trials = 100;
  abs_pose_options.ransac_options.max_num_trials = 10000;
  abs_pose_options.ransac_options.confidence = 0.99999;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  if (num_reg_images_per_camera_[image.CameraId()] > 0) {
    // Camera already refined from another image with the same camera.
    if (camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param)) {
      // Previously refined camera has bogus parameters,
      // so reset parameters and try to re-estimage.
      camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
      abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
      abs_pose_refinement_options.refine_focal_length = true;
      abs_pose_refinement_options.refine_extra_params = true;
    } else {
      abs_pose_options.estimate_focal_length = false;
      abs_pose_refinement_options.refine_focal_length = false;
      abs_pose_refinement_options.refine_extra_params = false;
    }
  } else {
    // Camera not refined before. Note that the camera parameters might have
    // been changed before but the image was filtered, so we explicitly reset
    // the camera parameters and try to re-estimate them.
    camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
    abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
    abs_pose_refinement_options.refine_focal_length = true;
    abs_pose_refinement_options.refine_extra_params = true;
  }

  if (!options.abs_pose_refine_focal_length) {
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
  }

  if (!options.abs_pose_refine_extra_params) {
    abs_pose_refinement_options.refine_extra_params = false;
  }

  size_t num_inliers;
  std::vector<char> inlier_mask;

  if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
                            &image.Qvec(), &image.Tvec(), &camera, &num_inliers,
                            &inlier_mask)) {
    return false;
  }

  if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Pose refinement
  //////////////////////////////////////////////////////////////////////////////

  if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
                          tri_points2D, tri_points3D, &image.Qvec(),
                          &image.Tvec(), &camera)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Continue tracks
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id);
  RegisterImageEvent(image_id);

  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const point2D_t point2D_idx = tri_corrs[i].first;
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) {
        const point3D_t point3D_id = tri_corrs[i].second;
        const TrackElement track_el(image_id, point2D_idx);
        reconstruction_->AddObservation(point3D_id, track_el);
        triangulator_->AddModifiedPoint3D(point3D_id);
      }
    }
  }

  return true;
}

size_t IncrementalMapper::TriangulateImage(
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t IncrementalMapper::Retriangulate(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper::CompleteTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper::MergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->MergeAllTracks(tri_options);
}

/**
 * 局部束调整函数 - 对新添加的图像及其相关图像进行局部优化
 * 
 * 该函数在增量式SfM重建过程中，为新注册的图像执行局部束调整，以优化相机参数和3D点坐标
 * 同时支持与激光雷达点云数据的融合约束
 * 
 * @param options 增量式重建的全局选项
 * @param ba_options 束调整的具体参数设置
 * @param tri_options 三角化的参数设置
 * @param image_id 当前新注册的图像ID
 * @param point3D_ids 需要调整的3D点ID集合
 * @return 局部束调整的结果报告，包含各类统计信息
 */
IncrementalMapper::LocalBundleAdjustmentReport
IncrementalMapper::AdjustLocalBundle(
    const Options& options, 
    const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const image_t image_id,
    const std::unordered_set<point3D_t>& point3D_ids) {
  // 确保重建对象已初始化和选项合法性
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  // 初始化报告结构
  LocalBundleAdjustmentReport report;

  // 查找与当前图像共享最多3D点的相关图像集合
  const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);

  std::cout<<std::endl;
  // 只有当找到相关图像时才执行束调整
  if (local_bundle.size() > 0) {
    // 创建束调整配置对象
    BundleAdjustmentConfig ba_config;
    // 如果启用激光雷达约束，则添加点云数据
    if (ba_options.if_add_lidar_constraint || ba_options.if_add_lidar_corresponding){
      ba_config.AddPointcloud(lidar_pointcloud_process_);
    }

    // 将当前图像添加到优化配置中
    ba_config.AddImage(image_id);

    // 检查初始图像是否在局部束中
    bool if_first_image_exist = false;

    // 将局部束中的所有相关图像添加到优化配置中
    for (const image_t local_image_id : local_bundle) {
      if (local_image_id == options.init_image_id1){
          if_first_image_exist = true;
      }
      ba_config.AddImage(local_image_id);
    }
    // for (const image_t local_image_id : local_bundle) {
    //   ba_config.AddImage(local_image_id);
    // }

    // 当启用激光雷达约束，且初始图像存在，且注册图像数量少于阈值时，固定初始图像的位姿
    // 这有助于建立一个稳定的全局坐标系
    if (ba_options.if_add_lidar_constraint && 
        if_first_image_exist && 
        reconstruction_->NumRegImages() < options.first_image_fixed_frames){
      ba_config.SetConstantPose(options.init_image_id1);
    } 

    // 如果选项指定，固定已存在的图像位姿
    if (options.fix_existing_images) {
      for (const image_t local_image_id : local_bundle) {
        if (existing_image_ids_.count(local_image_id)) {
          ba_config.SetConstantPose(local_image_id);
        }
      }
    }

    // 统计每个相机模型对应的图像数量
    // 这用于决定哪些相机参数应该固定
    std::unordered_map<camera_t, size_t> num_images_per_camera;
    for (const image_t image_id : ba_config.Images()) {
      const Image& image = reconstruction_->Image(image_id);
      num_images_per_camera[image.CameraId()] += 1;
    }

    // 如果局部束中某相机的图像数少于该相机的已注册总图像数
    // 则固定该相机参数，避免过度优化导致不稳定
    for (const auto& camera_id_and_num_images_pair : num_images_per_camera) {
      const size_t num_reg_images_for_camera =
          num_reg_images_per_camera_.at(camera_id_and_num_images_pair.first);
      if (camera_id_and_num_images_pair.second < num_reg_images_for_camera) {
        ba_config.SetConstantCamera(camera_id_and_num_images_pair.first);
      }
    }

    // 固定7自由度(7 DoF)，防止束调整中的尺度/旋转/平移漂移
    // 只有在不使用激光雷达约束时才需要，因为激光雷达点云已提供了尺度信息
    if (!ba_options.if_add_lidar_constraint) {
      if (local_bundle.size() == 1) {
        // 如果只有一个相关图像，固定其位姿并固定当前图像的一个平移分量
        ba_config.SetConstantPose(local_bundle[0]);
        ba_config.SetConstantTvec(image_id, {0});
      } else if (local_bundle.size() > 1) {
        // 如果有多个相关图像，固定最后一个图像的位姿
        // 并固定倒数第二个图像的一个平移分量(除非它是已存在的需要固定的图像)
        const image_t image_id1 = local_bundle[local_bundle.size() - 1];
        const image_t image_id2 = local_bundle[local_bundle.size() - 2];
        ba_config.SetConstantPose(image_id1);
        if (!options.fix_existing_images || 
            !existing_image_ids_.count(image_id2)) {
          ba_config.SetConstantTvec(image_id2, {0});
        }
      }
    }

    // 收集需要优化的3D点集合
    // 针对新的和短轨迹的3D点进行优化，长轨迹点通常已经很稳定，不需要频繁优化
    std::unordered_set<point3D_t> variable_point3D_ids;
    std::unordered_set<point3D_t> pcdproj_point3D_ids;
    std::unordered_set<point3D_t> search_closest_point3D_ids;

    // 如果启用激光雷达约束
    if (ba_options.if_add_lidar_constraint) {
      for (const point3D_t point3D_id : point3D_ids) {
        const Point3D& point3D = reconstruction_->Point3D(point3D_id);
        // 轨迹长度阈值较大，因为激光雷达约束下更多点需要优化
        const size_t kMaxTrackLength = 1000;
        if (!point3D.HasError() || point3D.Track().Length() <= kMaxTrackLength) {
          ba_config.AddVariablePoint(point3D_id);
          variable_point3D_ids.insert(point3D_id);
          // 根据轨迹长度分类点，短轨迹点用于投影匹配，长轨迹点用于最近点搜索
          if (point3D.Track().Length() < options.min_proj_num + 3){
            pcdproj_point3D_ids.insert(point3D_id);
          } else {
            search_closest_point3D_ids.insert(point3D_id);
          }
        }
      }
    } else {
      // 不使用激光雷达约束时，使用较小的轨迹长度阈值
      for (const point3D_t point3D_id : point3D_ids) {
        const Point3D& point3D = reconstruction_->Point3D(point3D_id);
        const size_t kMaxTrackLength = 15;
        if (!point3D.HasError() || point3D.Track().Length() <= kMaxTrackLength) {
          ba_config.AddVariablePoint(point3D_id);
          variable_point3D_ids.insert(point3D_id);
        }
      }
    }

    // 如果启用激光雷达约束或对应关系
    if (ba_options.if_add_lidar_constraint || ba_options.if_add_lidar_corresponding){
      // 处理短轨迹点：先投影到图像，再匹配到激光雷达点
      for (auto iter = pcdproj_point3D_ids.begin(); iter != pcdproj_point3D_ids.end(); iter++){
        // 获取短轨迹点ID
        const point3D_t point3D_id = *iter;
        // 设置特征匹配阈值
        int threshold = ba_options.ba_match_features_threshold;
        // 投影到图像
        ba_config.Project2Image(reconstruction_,point3D_id, image_id, threshold);
      }
      // 处理短轨迹点：匹配到激光雷达点
      for (auto iter = pcdproj_point3D_ids.begin(); iter != pcdproj_point3D_ids.end(); iter++){
        const point3D_t point3D_id = *iter;
        ba_config.MatchVariablePoint2LidarPoint(reconstruction_,point3D_id);
      }

      // 处理长轨迹点：直接搜索最近的激光雷达点
      for (auto iter = search_closest_point3D_ids.begin(); iter != search_closest_point3D_ids.end(); iter++){
        const point3D_t point3D_id = *iter;
        // const Point3D& point3D = reconstruction_->Point3D(point3D_id);
        // 根据点被优化次数动态调整搜索范围
        int opt_num = reconstruction_->Point3D(point3D_id).GlobalOptNum();
        double max_search_range = options.kdtree_max_search_range - opt_num * options.search_range_drop_speed;
        if (max_search_range <= options.kdtree_min_search_range) {
          max_search_range = options.kdtree_min_search_range;
        }
        ba_config.MatchClosestLidarPoint(reconstruction_,point3D_id,max_search_range);
      }
    }

    
    // 执行局部束调整
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    const BundleAdjuster::OptimazePhrase phrase = BundleAdjuster::OptimazePhrase::Local;
    bundle_adjuster.SetOptimazePhrase(phrase);
    bundle_adjuster.Solve(reconstruction_);

    // 记录调整的观测点数量
    report.num_adjusted_observations =
        bundle_adjuster.Summary().num_residuals / 2;

    // 合并优化后的轨迹与其他现有点
    report.num_merged_observations =
        triangulator_->MergeTracks(tri_options, variable_point3D_ids);
    // 完成之前可能因相机参数不准确而失败的轨迹三角化
    // 这有助于避免一些点被过滤掉，并有助于后续图像注册
    report.num_completed_observations =
        triangulator_->CompleteTracks(tri_options, variable_point3D_ids);
    report.num_completed_observations +=
        triangulator_->CompleteImage(tri_options, image_id);
  }

  // 过滤修改后的图像和所有变化的3D点，确保模型中没有离群点
  // 虽然这会导致重复工作(因为许多3D点可能同时存在于多个调整的图像中)
  // 但在此阶段过滤不是性能瓶颈
  std::unordered_set<image_t> filter_image_ids;
  filter_image_ids.insert(image_id);
  filter_image_ids.insert(local_bundle.begin(), local_bundle.end());

  // 过滤图像中的3D点，根据重投影误差和三角化角度
  report.num_filtered_observations = reconstruction_->FilterPoints3DInImages(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      filter_image_ids);

  // 过滤指定的3D点集合
  report.num_filtered_observations += reconstruction_->FilterPoints3D(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      point3D_ids);

  // 如果启用激光雷达约束，还需要过滤激光雷达离群点
  if (ba_options.if_add_lidar_constraint) {
    report.num_filtered_observations += reconstruction_->FilterLidarOutlier(
        options.proj_max_dist_error,options.icp_max_dist_error);
  }
  return report;
}

/**
 * 全局束调整函数 - 对整个重建场景进行全局优化
 * 
 * 该函数在增量式SfM重建过程中阶段性地执行全局束调整，优化所有相机参数和3D点坐标，
 * 以减小累积误差，提高整体重建精度。
 * 
 * @param options 增量式重建的全局选项
 * @param ba_options 束调整的具体参数设置
 * @return 优化是否成功
 */
bool IncrementalMapper::AdjustGlobalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
  // 确保重建对象已初始化
  CHECK_NOTNULL(reconstruction_);

  // 获取所有已注册图像的ID列表
  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  // 确保至少有两个已注册图像，这是全局束调整的最低要求
  CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                       "registered for global "
                                       "bundle-adjustment";

  // 过滤具有负深度的观测点，避免束调整中出现退化情况
  // 负深度表示点位于相机后方，这在物理上是不可能的
  reconstruction_->FilterObservationsWithNegativeDepth();

  // 创建并配置束调整参数
  BundleAdjustmentConfig ba_config;
  // 将所有已注册图像添加到优化配置中
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // 如果选项指定，固定已存在的图像位姿
  // 这通常用于增量式重建中，避免破坏已经重建好的部分
  if (options.fix_existing_images) {
    for (const image_t image_id : reg_image_ids) {
      if (existing_image_ids_.count(image_id)) {
        ba_config.SetConstantPose(image_id);
      }
    }
  }

  // 固定7个自由度，解决全局束调整中的规模/旋转/平移不确定性问题
  // 通常做法是固定第一个图像的所有位姿参数(6自由度)和第二个图像的一个平移分量(1自由度)
  ba_config.SetConstantPose(reg_image_ids[0]); // 固定第一个图像的位姿(旋转和平移)
  if (!options.fix_existing_images ||
      !existing_image_ids_.count(reg_image_ids[1])) {
    // 如果第二个图像不是已存在的需要固定的图像，则固定其一个平移分量
    ba_config.SetConstantTvec(reg_image_ids[1], {0}); // 固定第二个图像的X方向平移
  }

  // 创建并执行束调整
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  // 设置优化阶段为全局优化(区别于局部优化)
  const BundleAdjuster::OptimazePhrase phrase = BundleAdjuster::OptimazePhrase::Global;
  bundle_adjuster.SetOptimazePhrase(phrase);

  // 执行优化求解，如果失败则返回false
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  // 对场景进行归一化处理，以保持数值稳定性
  // 同时避免在可视化时出现过大的尺度变化
  // 归一化通常包括将场景中心移到坐标原点，并调整尺度使点云更加紧凑
  reconstruction_->Normalize();

  return true;
}

/**
 * 基于激光雷达约束的全局束调整函数
 * 
 * 该函数结合激光雷达点云数据对整个重建场景进行全局优化，提高几何精度和尺度准确性。
 * 与普通全局束调整不同，它允许使用激光雷达数据作为额外约束，并采用球形局部化策略。
 * 
 * @param options 增量式重建的全局选项
 * @param ba_options 束调整的参数设置，包含激光雷达相关选项
 * @return 优化是否成功
 */
bool IncrementalMapper::AdjustGlobalBundleByLidar(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
  // 检查重建对象是否已初始化
  CHECK_NOTNULL(reconstruction_);

  // 获取所有已注册图像的ID列表和所有3D点
  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();
  // 获取所有3D点的ID和对应的3D点数据
  EIGEN_STL_UMAP(point3D_t, Point3D) point3d_ids = reconstruction_->Points3D();

  // 确保至少有两个已注册图像，这是全局束调整的最低要求
  CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                       "registered for global "
                                       "bundle-adjustment";

  // 过滤具有负深度的观测点，避免束调整中出现退化情况
  reconstruction_->FilterObservationsWithNegativeDepth();

  // 创建束调整配置
  BundleAdjustmentConfig ba_config;
  // 将所有已注册图像添加到优化配置中
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // 如果选项指定，固定已存在的图像位姿
  if (options.fix_existing_images) {
    for (const image_t image_id : reg_image_ids) {
      if (existing_image_ids_.count(image_id)) {
        ba_config.SetConstantPose(image_id);
      }
    }
  }

  // 注释掉的代码是标准束调整中固定7自由度的方法
  // 在激光雷达约束下不需要，因为激光雷达提供了尺度和坐标系
  // ba_config.SetConstantPose(reg_image_ids[0]);
  // if (!options.fix_existing_images ||
  //     !existing_image_ids_.count(reg_image_ids[1])) {
  //   ba_config.SetConstantTvec(reg_image_ids[1], {0});
  // }

  // 如果已注册图像数量少于阈值，固定初始图像位姿
  // 这有助于在早期阶段保持坐标系稳定
  int num = reg_image_ids.size() - 1 ;
  if (num < options.first_image_fixed_frames){
    ba_config.SetConstantPose(options.init_image_id1);
    num +=1;
  }
    
  // 实现球形局部化策略：只优化最新图像周围特定半径内的图像和点
  // 获取最新注册图像的位置
  image_t latest_image_id = reg_image_ids.back();
  Eigen::Quaterniond latest_q_cw(reconstruction_->Image(latest_image_id).Qvec()[0],
                            reconstruction_->Image(latest_image_id).Qvec()[1],
                            reconstruction_->Image(latest_image_id).Qvec()[2],
                            reconstruction_->Image(latest_image_id).Qvec()[3]);
  Eigen::Matrix3d latest_rot_cw = latest_q_cw.toRotationMatrix();
  Eigen::Vector3d latest_t_cw = reconstruction_->Image(latest_image_id).Tvec();
  // 计算最新图像的世界坐标(相机中心位置)
  Eigen::Vector3d latest_image_T = - latest_rot_cw.transpose() * latest_t_cw;

  // 用于存储球内和球外的图像
  std::vector<image_t> image_in_sphere; // 球内图像，会被优化
  std::vector<image_t> image_out_sphere; // 球外图像，位姿将被固定
  std::unordered_set<point3D_t> variable_point3D_ids; // 需要优化的3D点ID集合

  // 遍历所有已注册图像，根据与最新图像的距离判断是否在球内
  for (const image_t& image_id : reg_image_ids){
    // 计算当前图像的世界坐标
    Eigen::Quaterniond q_cw(reconstruction_->Image(image_id).Qvec()[0],
                            reconstruction_->Image(image_id).Qvec()[1],
                            reconstruction_->Image(image_id).Qvec()[2],
                            reconstruction_->Image(image_id).Qvec()[3]);
    Eigen::Matrix3d rot_cw = q_cw.toRotationMatrix();
    Eigen::Vector3d t_cw = reconstruction_->Image(image_id).Tvec();
    Eigen::Vector3d image_T = - rot_cw.transpose() * t_cw;

    // 计算与最新图像的距离
    double dist = (latest_image_T - image_T).norm();
    // 根据距离将图像分为球内和球外
    if (dist <= options.ba_spherical_search_radius) {
      image_in_sphere.push_back(image_id);
    } else {
      image_out_sphere.push_back(image_id);
    }
  }

  // 固定球外图像的位姿，这些图像不参与优化
  for (const image_t image_id : image_out_sphere) {
    ba_config.SetConstantPose(image_id);
  }

  // 收集球内图像观测到的所有3D点，这些点需要被优化
  for (image_t image_id : image_in_sphere) {
    std::vector<class Point2D> point2Ds = reconstruction_->Image(image_id).Points2D();
    for (Point2D& point2D : point2Ds) {
      // 跳过没有对应3D点的2D特征点
      if (!point2D.HasPoint3D()) {
            continue;
      }
      // 获取对应的3D点ID
      point3D_t point3d_id = point2D.GetPoint3DId();
      // 确认该3D点存在于重建中
      auto iter = point3d_ids.find(point3d_id);
      if (iter != point3d_ids.end()){
        // 将该点添加为可变点(需要优化)
        ba_config.AddVariablePoint(point3d_id);
        variable_point3D_ids.insert(point3d_id);
      }
    }
  }

  // 如果启用激光雷达约束
  if (ba_options.if_add_lidar_constraint || ba_options.if_add_lidar_corresponding){
    // 为每个需要优化的3D点查找对应的激光雷达点
    for (auto iter = variable_point3D_ids.begin(); iter != variable_point3D_ids.end(); iter++){
      point3D_t point3D_id = *iter;
      Point3D& point3D = reconstruction_->Point3D(point3D_id);
      // 标记该点在优化球内
      point3D.IfInSphere() = true;
      // int track_length = point3D.Track().Length();
      // double max_search_range = options.kdtree_max_search_range - (track_length - 3) * options.search_range_drop_speed;

      // 根据点被优化的次数动态调整搜索范围
      // 优化次数越多，搜索范围越小，表示对该点位置越有信心
      int opt_num = point3D.GlobalOptNum();
      double max_search_range = options.kdtree_max_search_range - opt_num * options.search_range_drop_speed;
      if (max_search_range <= options.kdtree_min_search_range) {
        max_search_range = options.kdtree_min_search_range;
      }

      // 获取3D点坐标并在激光雷达点云中搜索最近点
      Eigen::Vector3d pt_xyz = point3D.XYZ();
      Eigen::Vector6d lidar_pt; // 包含位置和法向量的激光雷达点
      if (lidar_pointcloud_process_->SearchNearestNeiborByKdtree(pt_xyz,lidar_pt)) {
        // 提取法向量和点坐标
        Eigen::Vector3d norm = lidar_pt.block(3,0,3,1); // 法向量
        Eigen::Vector3d l_pt = lidar_pt.block(0,0,3,1); // 点坐标

        // 计算平面方程 ax+by+cz+d=0 中的d参数
        double d = 0 - l_pt.dot(norm);
        Eigen::Vector4d plane;
        plane << norm(0),norm(1),norm(2),d;

        // 创建激光雷达点对象
        LidarPoint lidar_point(l_pt,plane);

        // 根据法向量判断点的类型(地面点或普通点)
        // 如果y方向法向量远大于x和z方向，认为是地面点
        if (std::abs(norm(1)/norm(0))>10 && std::abs(norm(1)/norm(2))>10) {
          lidar_point.SetType(LidarPointType::IcpGround); // 设置为地面点
          // 设置地面点颜色为黄色
          Eigen::Vector3ub color;
          color << 255,255,0;
          lidar_point.SetColor(color);
        } else {
          lidar_point.SetType(LidarPointType::Icp); // 设置为普通点
          // 设置普通点颜色为蓝色
          Eigen::Vector3ub color;
          color << 0,0,255;
          lidar_point.SetColor(color);
        }

        // 计算点到点距离，如果超出搜索范围则跳过
        double dist = lidar_point.ComputePointToPointDist(pt_xyz);
        if (dist > max_search_range) continue;

        // 将激光雷达点添加到束调整配置和重建中
        ba_config.AddLidarPoint(point3D_id,lidar_point);
        reconstruction_ -> AddLidarPointInGlobal(point3D_id,lidar_point);
      }
    }
  }
  
  // 创建并执行束调整
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  const BundleAdjuster::OptimazePhrase phrase = BundleAdjuster::OptimazePhrase::Global;
  bundle_adjuster.SetOptimazePhrase(phrase);

  // 执行优化求解，如果失败则返回false
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  // 更新所有优化过的3D点
  for (auto iter = variable_point3D_ids.begin(); iter != variable_point3D_ids.end(); iter++) {
    Point3D& Point3D = reconstruction_ -> Point3D(*iter);
    Point3D.AddGlobalOptNum(); // 增加全局优化计数
    Point3D.IfInSphere() = false; // 重置球内标记
  }

  // 注释掉的归一化步骤，在激光雷达约束下通常不需要归一化，因为已有真实尺度
  // reconstruction_->Normalize();

  return true;
}

/**
 * 并行全局束调整函数 - 使用并行计算加速整个场景的优化
 * 
 * 该函数提供了一个并行计算版本的全局束调整，适用于大规模重建场景，
 * 能够利用多核处理器加速计算过程，提高大场景优化效率。
 * 
 * @param ba_options 常规束调整的参数设置
 * @param parallel_ba_options 并行束调整的特有参数设置
 * @return 优化是否成功
 */
bool IncrementalMapper::AdjustParallelGlobalBundle(
    const BundleAdjustmentOptions& ba_options,
    const ParallelBundleAdjuster::Options& parallel_ba_options) {
  // 确保重建对象已初始化
  CHECK_NOTNULL(reconstruction_);

  // 获取所有已注册图像的ID列表
  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  // 确保至少有两个已注册图像，这是全局束调整的最低要求
  CHECK_GE(reg_image_ids.size(), 2)
      << "At least two images must be registered for global bundle-adjustment";

  // 过滤具有负深度的观测点，避免束调整中出现退化情况
  // 负深度表示点位于相机后方，这在物理上是不可能的
  reconstruction_->FilterObservationsWithNegativeDepth();

  // 创建并配置束调整参数
  BundleAdjustmentConfig ba_config;
  // 将所有已注册图像添加到优化配置中
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // 注意：与普通全局束调整不同，这里没有显式固定7个自由度
  // 并行束调整器可能内部处理了这个问题，或者使用了其他约束方法

  // 创建并执行并行束调整
  // 使用专门的ParallelBundleAdjuster类，它能够利用多线程加速优化过程
  ParallelBundleAdjuster bundle_adjuster(parallel_ba_options, ba_options,
                                         ba_config);

  // 执行优化求解，如果失败则返回false
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  // 对场景进行归一化处理，以保持数值稳定性
  // 同时避免在可视化时出现过大的尺度变化
  // 归一化通常包括将场景中心移到坐标原点，并调整尺度使点云更加紧凑
  reconstruction_->Normalize();

  // 优化成功，返回true
  return true;
}

size_t IncrementalMapper::FilterImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  // Do not filter images in the early stage of the reconstruction, since the
  // calibration is often still refining a lot. Hence, the camera parameters
  // are not stable in the beginning.
  const size_t kMinNumImages = 20;
  if (reconstruction_->NumRegImages() < kMinNumImages) {
    return {};
  }

  const std::vector<image_t> image_ids = reconstruction_->FilterImages(
      options.min_focal_length_ratio, options.max_focal_length_ratio,
      options.max_extra_param);

  for (const image_t image_id : image_ids) {
    DeRegisterImageEvent(image_id);
    filtered_images_.insert(image_id);
  }

  return image_ids.size();
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());
  return reconstruction_->FilterAllPoints3D(options.filter_max_reproj_error,
                                            options.filter_min_tri_angle);
}

const Reconstruction& IncrementalMapper::GetReconstruction() const {
  CHECK_NOTNULL(reconstruction_);
  return *reconstruction_;
}

size_t IncrementalMapper::NumTotalRegImages() const {
  return num_total_reg_images_;
}

size_t IncrementalMapper::NumSharedRegImages() const {
  return num_shared_reg_images_;
}

const std::unordered_set<point3D_t>& IncrementalMapper::GetModifiedPoints3D() {
  return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper::ClearModifiedPoints3D() {
  triangulator_->ClearModifiedPoints3D();
}
void IncrementalMapper::ClearLidarPoints(){
  reconstruction_->ClearLidarPoints();
  reconstruction_->ClearLidarPointsInGlobal();
}
std::vector<image_t> IncrementalMapper::FindFirstInitialImage(
    const Options& options) const {
  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    image_t num_correspondences;
  };

  const size_t init_max_reg_trials =
      static_cast<size_t>(options.init_max_reg_trials);

  // Collect information of all not yet registered images with
  // correspondences.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto& image : reconstruction_->Images()) {
    // Only images with correspondences can be registered.
    if (image.second.NumCorrespondences() == 0) {
      continue;
    }

    // Only use images for initialization a maximum number of times.
    if (init_num_reg_trials_.count(image.first) &&
        init_num_reg_trials_.at(image.first) >= init_max_reg_trials) {
      continue;
    }

    // Only use images for initialization that are not registered in any
    // of the other reconstructions.
    if (num_registrations_.count(image.first) > 0 &&
        num_registrations_.at(image.first) > 0) {
      continue;
    }

    const class Camera& camera =
        reconstruction_->Camera(image.second.CameraId());
    ImageInfo image_info;
    image_info.image_id = image.first;
    image_info.prior_focal_length = camera.HasPriorFocalLength();
    image_info.num_correspondences = image.second.NumCorrespondences();
    image_infos.push_back(image_info);
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> IncrementalMapper::FindSecondInitialImage(
    const Options& options, const image_t image_id1) const {
  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();

  // Collect images that are connected to the first seed image and have
  // not been registered before in other reconstructions.
  const class Image& image1 = reconstruction_->Image(image_id1);
  std::unordered_map<image_t, point2D_t> num_correspondences;
  for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
       ++point2D_idx) {
    for (const auto& corr :
         correspondence_graph.FindCorrespondences(image_id1, point2D_idx)) {
      if (num_registrations_.count(corr.image_id) == 0 ||
          num_registrations_.at(corr.image_id) == 0) {
        num_correspondences[corr.image_id] += 1;
      }
    }
  }

  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    point2D_t num_correspondences;
  };

  const size_t init_min_num_inliers =
      static_cast<size_t>(options.init_min_num_inliers);

  // Compose image information in a compact form for sorting.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto elem : num_correspondences) {
    if (elem.second >= init_min_num_inliers) {
      const class Image& image = reconstruction_->Image(elem.first);
      const class Camera& camera = reconstruction_->Camera(image.CameraId());
      ImageInfo image_info;
      image_info.image_id = elem.first;
      image_info.prior_focal_length = camera.HasPriorFocalLength();
      image_info.num_correspondences = elem.second;
      image_infos.push_back(image_info);
    }
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> IncrementalMapper::FindLocalBundle(
    const Options& options, const image_t image_id) const {
  CHECK(options.Check());

  const Image& image = reconstruction_->Image(image_id);
  CHECK(image.IsRegistered());

  // Extract all images that have at least one 3D point with the query image
  // in common, and simultaneously count the number of common 3D points.

  std::unordered_map<image_t, size_t> shared_observations;

  std::unordered_set<point3D_t> point3D_ids;
  point3D_ids.reserve(image.NumPoints3D());

  for (const Point2D& point2D : image.Points2D()) {
    if (point2D.HasPoint3D()) {
      point3D_ids.insert(point2D.Point3DId());
      const Point3D& point3D = reconstruction_->Point3D(point2D.Point3DId());
      for (const TrackElement& track_el : point3D.Track().Elements()) {
        if (track_el.image_id != image_id) {
          shared_observations[track_el.image_id] += 1;
        }
      }
    }
  }

  // Sort overlapping images according to number of shared observations.

  std::vector<std::pair<image_t, size_t>> overlapping_images(
      shared_observations.begin(), shared_observations.end());
  std::sort(overlapping_images.begin(), overlapping_images.end(),
            [](const std::pair<image_t, size_t>& image1,
               const std::pair<image_t, size_t>& image2) {
              return image1.second > image2.second;
            });

  // The local bundle is composed of the given image and its most connected
  // neighbor images, hence the subtraction of 1.

  const size_t num_images =
      static_cast<size_t>(options.local_ba_num_images - 1);
  const size_t num_eff_images = std::min(num_images, overlapping_images.size());

  // Extract most connected images and ensure sufficient triangulation angle.

  std::vector<image_t> local_bundle_image_ids;
  local_bundle_image_ids.reserve(num_eff_images);

  // If the number of overlapping images equals the number of desired images in
  // the local bundle, then simply copy over the image identifiers.
  if (overlapping_images.size() == num_eff_images) {
    for (const auto& overlapping_image : overlapping_images) {
      local_bundle_image_ids.push_back(overlapping_image.first);
    }
    return local_bundle_image_ids;
  }

  // In the following iteration, we start with the most overlapping images and
  // check whether it has sufficient triangulation angle. If none of the
  // overlapping images has sufficient triangulation angle, we relax the
  // triangulation angle threshold and start from the most overlapping image
  // again. In the end, if we still haven't found enough images, we simply use
  // the most overlapping images.

  const double min_tri_angle_rad = DegToRad(options.local_ba_min_tri_angle);

  // The selection thresholds (minimum triangulation angle, minimum number of
  // shared observations), which are successively relaxed.
  const std::array<std::pair<double, double>, 8> selection_thresholds = {{
      std::make_pair(min_tri_angle_rad / 1.0, 0.6 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 1.5, 0.6 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 2.0, 0.5 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 2.5, 0.4 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 3.0, 0.3 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 4.0, 0.2 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 5.0, 0.1 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 6.0, 0.1 * image.NumPoints3D()),
  }};

  const Eigen::Vector3d proj_center = image.ProjectionCenter();
  std::vector<Eigen::Vector3d> shared_points3D;
  shared_points3D.reserve(image.NumPoints3D());
  std::vector<double> tri_angles(overlapping_images.size(), -1.0);
  std::vector<char> used_overlapping_images(overlapping_images.size(), false);

  for (const auto& selection_threshold : selection_thresholds) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Check if the image has sufficient overlap. Since the images are ordered
      // based on the overlap, we can just skip the remaining ones.
      if (overlapping_images[overlapping_image_idx].second <
          selection_threshold.second) {
        break;
      }

      // Check if the image is already in the local bundle.
      if (used_overlapping_images[overlapping_image_idx]) {
        continue;
      }

      const auto& overlapping_image = reconstruction_->Image(
          overlapping_images[overlapping_image_idx].first);
      const Eigen::Vector3d overlapping_proj_center =
          overlapping_image.ProjectionCenter();

      // In the first iteration, compute the triangulation angle. In later
      // iterations, reuse the previously computed value.
      double& tri_angle = tri_angles[overlapping_image_idx];
      if (tri_angle < 0.0) {
        // Collect the commonly observed 3D points.
        shared_points3D.clear();
        for (const Point2D& point2D : image.Points2D()) {
          if (point2D.HasPoint3D() && point3D_ids.count(point2D.Point3DId())) {
            shared_points3D.push_back(
                reconstruction_->Point3D(point2D.Point3DId()).XYZ());
          }
        }

        // Calculate the triangulation angle at a certain percentile.
        const double kTriangulationAnglePercentile = 75;
        tri_angle = Percentile(
            CalculateTriangulationAngles(proj_center, overlapping_proj_center,
                                         shared_points3D),
            kTriangulationAnglePercentile);
      }

      // Check that the image has sufficient triangulation angle.
      if (tri_angle >= selection_threshold.first) {
        local_bundle_image_ids.push_back(overlapping_image.ImageId());
        used_overlapping_images[overlapping_image_idx] = true;
        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }

    // Check if we already collected enough images.
    if (local_bundle_image_ids.size() >= num_eff_images) {
      break;
    }
  }

  // In case there are not enough images with sufficient triangulation angle,
  // simply fill up the rest with the most overlapping images.

  if (local_bundle_image_ids.size() < num_eff_images) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Collect image if it is not yet in the local bundle.
      if (!used_overlapping_images[overlapping_image_idx]) {
        local_bundle_image_ids.push_back(
            overlapping_images[overlapping_image_idx].first);
        used_overlapping_images[overlapping_image_idx] = true;

        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }
  }

  return local_bundle_image_ids;
}

void IncrementalMapper::RegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_[image.CameraId()];
  num_reg_images_for_camera += 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image += 1;
  if (num_regs_for_image == 1) {
    num_total_reg_images_ += 1;
  } else if (num_regs_for_image > 1) {
    num_shared_reg_images_ += 1;
  }
}

void IncrementalMapper::DeRegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_.at(image.CameraId());
  CHECK_GT(num_reg_images_for_camera, 0);
  num_reg_images_for_camera -= 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image -= 1;
  if (num_regs_for_image == 0) {
    num_total_reg_images_ -= 1;
  } else if (num_regs_for_image > 0) {
    num_shared_reg_images_ -= 1;
  }
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(
    const Options& options, const image_t image_id1, const image_t image_id2) {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);

  if (prev_init_image_pair_id_ == image_pair_id) {
    return true;
  }

  const Image& image1 = database_cache_->Image(image_id1);
  const Camera& camera1 = database_cache_->Camera(image1.CameraId());

  const Image& image2 = database_cache_->Image(image_id2);
  const Camera& camera2 = database_cache_->Camera(image2.CameraId());

  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  const FeatureMatches matches =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  std::vector<Eigen::Vector2d> points1;
  points1.reserve(image1.NumPoints2D());
  for (const auto& point : image1.Points2D()) {
    points1.push_back(point.XY());
  }

  std::vector<Eigen::Vector2d> points2;
  points2.reserve(image2.NumPoints2D());
  for (const auto& point : image2.Points2D()) {
    points2.push_back(point.XY());
  }

  TwoViewGeometry two_view_geometry;
  TwoViewGeometry::Options two_view_geometry_options;
  two_view_geometry_options.ransac_options.min_num_trials = 30;
  two_view_geometry_options.ransac_options.max_error = options.init_max_error;
  // Estimate E,F,H
  two_view_geometry.EstimateCalibrated(camera1, points1, camera2, points2,
                                       matches, two_view_geometry_options);
  // Estimate relative pose
  if (!two_view_geometry.EstimateRelativePose(camera1, points1, camera2,
                                              points2)) {
    return false;
  }

  if (static_cast<int>(two_view_geometry.inlier_matches.size()) >=
          options.init_min_num_inliers &&
      std::abs(two_view_geometry.tvec.z()) < options.init_max_forward_motion &&
      two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
    prev_init_image_pair_id_ = image_pair_id;
    prev_init_two_view_geometry_ = two_view_geometry;
    return true;
  }

  return false;
}

}  // namespace colmap
