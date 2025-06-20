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

#include "sfm/incremental_triangulator.h"

#include "base/projection.h"
#include "estimators/triangulation.h"
#include "util/misc.h"

namespace colmap {

bool IncrementalTriangulator::Options::Check() const {
  CHECK_OPTION_GE(max_transitivity, 0);
  CHECK_OPTION_GT(create_max_angle_error, 0);
  CHECK_OPTION_GT(continue_max_angle_error, 0);
  CHECK_OPTION_GT(merge_max_reproj_error, 0);
  CHECK_OPTION_GT(complete_max_reproj_error, 0);
  CHECK_OPTION_GE(complete_max_transitivity, 0);
  CHECK_OPTION_GT(re_max_angle_error, 0);
  CHECK_OPTION_GE(re_min_ratio, 0);
  CHECK_OPTION_LE(re_min_ratio, 1);
  CHECK_OPTION_GE(re_max_trials, 0);
  CHECK_OPTION_GT(min_angle, 0);
  return true;
}

IncrementalTriangulator::IncrementalTriangulator(
    const CorrespondenceGraph* correspondence_graph,
    Reconstruction* reconstruction)
    : correspondence_graph_(correspondence_graph),
      reconstruction_(reconstruction) {}

/**
 * 对指定图像进行三角化处理
 * 
 * 这是增量式SfM重建中的核心函数，用于为新注册的图像创建3D点。
 * 函数会遍历图像中的所有2D特征点，寻找与其他已注册图像的对应关系，
 * 然后通过三角化创建新的3D点或将2D点关联到已有的3D点。
 * 
 * @param options 三角化选项，包含各种阈值和约束参数
 * @param image_id 要进行三角化的图像ID
 * @return 成功三角化的2D点数量（即新建或关联的观测数量）
 */
size_t IncrementalTriangulator::TriangulateImage(const Options& options,
                                                 const image_t image_id) {
  // 检查选项参数的有效性，确保所有阈值都在合理范围内
  CHECK(options.Check());

  // 记录本次三角化成功处理的2D点数量
  size_t num_tris = 0;

  // 清空各种缓存数据结构，确保使用最新的重建状态
  // 包括相机参数异常检查缓存、合并尝试记录、找到的对应关系缓存等
  ClearCaches();

  // 获取要处理的图像对象
  const Image& image = reconstruction_->Image(image_id);
  
  // 检查图像是否已经注册（即是否已经估计出相机位姿）
  // 只有已注册的图像才能参与三角化，因为需要已知的相机位姿
  if (!image.IsRegistered()) {
    return num_tris;  // 未注册的图像直接返回0
  }

  // 获取该图像对应的相机参数
  const Camera& camera = reconstruction_->Camera(image.CameraId());
  
  // 检查相机参数是否异常（如焦距过大/过小、畸变参数异常等）
  // 异常的相机参数会导致不可靠的三角化结果
  if (HasCameraBogusParams(options, camera)) {
    return num_tris;  // 相机参数异常的图像跳过处理
  }

  // 创建参考对应关系数据结构
  // 这个结构体包含了当前图像的完整信息，在后续处理中作为参考点
  CorrData ref_corr_data;
  ref_corr_data.image_id = image_id;     // 图像ID
  ref_corr_data.image = &image;          // 图像对象指针
  ref_corr_data.camera = &camera;       // 相机对象指针

  // 用于存储从参考观测到其他图像的对应关系
  // 每次处理一个2D点时，这里会存储所有与该点有对应关系的其他图像中的点
  std::vector<CorrData> corrs_data;

  // 遍历当前图像中的所有2D特征点，逐个尝试三角化
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    
    // 查找当前2D点在其他图像中的对应关系
    // Find函数会搜索与当前点匹配的所有其他图像中的特征点
    // max_transitivity参数控制搜索的传递深度（多跳匹配）
    const size_t num_triangulated =
        Find(options, image_id, point2D_idx,
             static_cast<size_t>(options.max_transitivity), &corrs_data);
    
    // 如果没有找到任何对应关系，跳过当前点
    if (corrs_data.empty()) {
      continue;
    }

    // 获取当前处理的2D点对象
    const Point2D& point2D = image.Point2D(point2D_idx);
    
    // 设置参考对应关系数据的具体2D点信息
    ref_corr_data.point2D_idx = point2D_idx;  // 2D点在图像中的索引
    ref_corr_data.point2D = &point2D;         // 2D点对象指针

    // 根据找到的对应关系中已三角化的点数量，采用不同的处理策略
    if (num_triangulated == 0) {
      // 情况1：所有对应点都未三角化
      // 这意味着我们需要创建全新的3D点
      
      // 将当前参考点也加入对应关系列表
      corrs_data.push_back(ref_corr_data);
      
      // 调用Create函数尝试创建新的3D点
      // Create会验证几何约束（如三角化角度），如果满足条件就创建3D点
      num_tris += Create(options, corrs_data);
    } else {
      // 情况2：部分对应点已经三角化
      // 这种情况下我们有两个任务：
      // 1. 尝试将当前点关联到已有的3D点（Continue操作）
      // 2. 为未三角化的对应点创建新的3D点（Create操作）
      
      // 首先尝试将当前参考点继续到已有的3D点轨迹中
      // Continue会检查当前点到已有3D点的重投影误差，如果足够小就建立关联
      num_tris += Continue(options, ref_corr_data, corrs_data);
      
      // 然后尝试为剩余未三角化的对应点创建新的3D点
      // 将参考点加入列表，因为它可能仍然没有被Continue操作处理
      corrs_data.push_back(ref_corr_data);
      num_tris += Create(options, corrs_data);
    }
  }

  // 返回本次三角化操作成功处理的2D点总数
  // 这个数字包括新创建的3D点观测和新关联到已有3D点的观测
  return num_tris;
}

size_t IncrementalTriangulator::CompleteImage(const Options& options,
                                              const image_t image_id) {
  CHECK(options.Check());

  size_t num_tris = 0;

  ClearCaches();

  const Image& image = reconstruction_->Image(image_id);
  if (!image.IsRegistered()) {
    return num_tris;
  }

  const Camera& camera = reconstruction_->Camera(image.CameraId());
  if (HasCameraBogusParams(options, camera)) {
    return num_tris;
  }

  // Setup estimation options.
  EstimateTriangulationOptions tri_options;
  tri_options.min_tri_angle = DegToRad(options.min_angle);
  tri_options.residual_type =
      TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
  tri_options.ransac_options.max_error = options.complete_max_reproj_error;
  tri_options.ransac_options.confidence = 0.9999;
  tri_options.ransac_options.min_inlier_ratio = 0.02;
  tri_options.ransac_options.max_num_trials = 10000;

  // Correspondence data for reference observation in given image. We iterate
  // over all observations of the image and each observation once becomes
  // the reference correspondence.
  CorrData ref_corr_data;
  ref_corr_data.image_id = image_id;
  ref_corr_data.image = &image;
  ref_corr_data.camera = &camera;

  // Container for correspondences from reference observation to other images.
  std::vector<CorrData> corrs_data;

  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);
    if (point2D.HasPoint3D()) {
      // Complete existing track.
      num_tris += Complete(options, point2D.Point3DId());
      continue;
    }

    if (options.ignore_two_view_tracks &&
        correspondence_graph_->IsTwoViewObservation(image_id, point2D_idx)) {
      continue;
    }

    const size_t num_triangulated =
        Find(options, image_id, point2D_idx,
             static_cast<size_t>(options.max_transitivity), &corrs_data);
    if (num_triangulated || corrs_data.empty()) {
      continue;
    }

    ref_corr_data.point2D = &point2D;
    ref_corr_data.point2D_idx = point2D_idx;
    corrs_data.push_back(ref_corr_data);

    // Setup data for triangulation estimation.
    std::vector<TriangulationEstimator::PointData> point_data;
    point_data.resize(corrs_data.size());
    std::vector<TriangulationEstimator::PoseData> pose_data;
    pose_data.resize(corrs_data.size());
    for (size_t i = 0; i < corrs_data.size(); ++i) {
      const CorrData& corr_data = corrs_data[i];
      point_data[i].point = corr_data.point2D->XY();
      point_data[i].point_normalized =
          corr_data.camera->ImageToWorld(point_data[i].point);
      pose_data[i].proj_matrix = corr_data.image->ProjectionMatrix();
      pose_data[i].proj_center = corr_data.image->ProjectionCenter();
      pose_data[i].camera = corr_data.camera;
    }

    // Enforce exhaustive sampling for small track lengths.
    const size_t kExhaustiveSamplingThreshold = 15;
    if (point_data.size() <= kExhaustiveSamplingThreshold) {
      tri_options.ransac_options.min_num_trials =
          NChooseK(point_data.size(), 2);
    }

    // Estimate triangulation.
    Eigen::Vector3d xyz;
    std::vector<char> inlier_mask;
    if (!EstimateTriangulation(tri_options, point_data, pose_data, &inlier_mask,
                               &xyz)) {
      continue;
    }

    // Add inliers to estimated track.
    Track track;
    track.Reserve(corrs_data.size());
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        const CorrData& corr_data = corrs_data[i];
        track.AddElement(corr_data.image_id, corr_data.point2D_idx);
        num_tris += 1;
      }
    }

    const point3D_t point3D_id =
        reconstruction_->AddPoint3D(xyz, std::move(track));
    modified_point3D_ids_.insert(point3D_id);
  }

  return num_tris;
}

size_t IncrementalTriangulator::CompleteTracks(
    const Options& options, const std::unordered_set<point3D_t>& point3D_ids) {
  CHECK(options.Check());

  size_t num_completed = 0;

  ClearCaches();

  for (const point3D_t point3D_id : point3D_ids) {
    num_completed += Complete(options, point3D_id);
  }

  return num_completed;
}

size_t IncrementalTriangulator::CompleteAllTracks(const Options& options) {
  CHECK(options.Check());

  size_t num_completed = 0;

  ClearCaches();

  for (const point3D_t point3D_id : reconstruction_->Point3DIds()) {
    num_completed += Complete(options, point3D_id);
  }

  return num_completed;
}

size_t IncrementalTriangulator::MergeTracks(
    const Options& options, const std::unordered_set<point3D_t>& point3D_ids) {
  CHECK(options.Check());

  size_t num_merged = 0;

  ClearCaches();

  for (const point3D_t point3D_id : point3D_ids) {
    num_merged += Merge(options, point3D_id);
  }

  return num_merged;
}

size_t IncrementalTriangulator::MergeAllTracks(const Options& options) {
  CHECK(options.Check());

  size_t num_merged = 0;

  ClearCaches();

  for (const point3D_t point3D_id : reconstruction_->Point3DIds()) {
    num_merged += Merge(options, point3D_id);
  }

  return num_merged;
}

/**
 * 重三角化函数 - 对已有重建进行补充三角化
 * 
 * 该函数是增量式SfM重建中的重要优化步骤，用于提高重建的完整性。
 * 它重新检查那些三角化率较低的图像对，尝试为更多的特征匹配创建3D点，
 * 从而增加重建的密度和鲁棒性。这对于处理纹理稀少或匹配困难的区域特别有效。
 * 
 * @param options 三角化选项，包含各种阈值和约束参数
 * @return 新创建的3D点数量
 */
size_t IncrementalTriangulator::Retriangulate(const Options& options) {
  // 检查选项参数的有效性
  CHECK(options.Check());

  // 记录本次重三角化创建的3D点总数
  size_t num_tris = 0;

  // 清空缓存，确保使用最新的重建状态
  // 这包括清空各种临时数据结构和统计信息
  ClearCaches();

  // 创建重三角化专用的选项配置
  // 重三角化使用更宽松的角度误差阈值，因为这是对已有重建的补充
  Options re_options = options;
  re_options.continue_max_angle_error = options.re_max_angle_error;

  // 遍历重建中的所有图像对
  // 图像对记录了两张图像之间的匹配统计信息
  for (const auto& image_pair : reconstruction_->ImagePairs()) {
    // 计算当前图像对的三角化比例
    // 三角化比例 = 已三角化的对应关系数量 / 总对应关系数量
    // Only perform retriangulation for under-reconstructed image pairs.
    const double tri_ratio =
        static_cast<double>(image_pair.second.num_tri_corrs) /
        static_cast<double>(image_pair.second.num_total_corrs);
    
    // 如果三角化比例已经足够高，跳过该图像对
    // 这样可以避免在已经充分重建的区域浪费计算资源
    if (tri_ratio >= options.re_min_ratio) {
      continue;
    }

    // 从图像对ID中提取两个图像的ID
    // Check if images are registered yet.
    image_t image_id1;
    image_t image_id2;
    Database::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);

    // 获取第一张图像并检查是否已注册
    // 只有已注册的图像才能参与重三角化，因为需要已知的相机位姿
    const Image& image1 = reconstruction_->Image(image_id1);
    if (!image1.IsRegistered()) {
      continue;
    }

    // 获取第二张图像并检查是否已注册
    const Image& image2 = reconstruction_->Image(image_id2);
    if (!image2.IsRegistered()) {
      continue;
    }

    // 限制每个图像对的最大重三角化尝试次数
    // 避免在困难的图像对上反复尝试，浪费计算资源
    // Only perform retriangulation for a maximum number of trials.
    int& num_re_trials = re_num_trials_[image_pair.first];
    if (num_re_trials >= options.re_max_trials) {
      continue;
    }
    num_re_trials += 1;  // 增加尝试次数计数器

    // 获取两张图像的相机参数
    const Camera& camera1 = reconstruction_->Camera(image1.CameraId());
    const Camera& camera2 = reconstruction_->Camera(image2.CameraId());
    
    // 检查相机参数是否异常
    // 异常的相机参数会导致不可靠的三角化结果
    if (HasCameraBogusParams(options, camera1) ||
        HasCameraBogusParams(options, camera2)) {
      continue;
    }

    // 查找两张图像之间的特征匹配关系
    // Find correspondences and perform retriangulation.
    const FeatureMatches& corrs =
        correspondence_graph_->FindCorrespondencesBetweenImages(image_id1,
                                                                image_id2);

    // 遍历所有特征匹配，尝试进行重三角化
    for (const auto& corr : corrs) {
      // 获取匹配的两个2D特征点
      const Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
      const Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);

      // 处理已有3D点的情况
      // 如果两个2D点都已经关联到3D点，需要区分两种情况：
      // 1. 关联到同一个3D点：无需处理，已经正确关联
      // 2. 关联到不同3D点：不进行重三角化，避免破坏现有结构
      // Two cases are possible here: both points belong to the same 3D point
      // or to different 3D points. In the former case, there is nothing
      // to do. In the latter case, we do not attempt retriangulation,
      // as retriangulated correspondences are very likely bogus and
      // would therefore destroy both 3D points if merged.
      if (point2D1.HasPoint3D() && point2D2.HasPoint3D()) {
        continue;
      }

      // 创建第一个对应关系数据结构
      // 包含图像、相机、2D点等完整信息，用于后续三角化
      CorrData corr_data1;
      corr_data1.image_id = image_id1;
      corr_data1.point2D_idx = corr.point2D_idx1;
      corr_data1.image = &image1;
      corr_data1.camera = &camera1;
      corr_data1.point2D = &point2D1;

      // 创建第二个对应关系数据结构
      CorrData corr_data2;
      corr_data2.image_id = image_id2;
      corr_data2.point2D_idx = corr.point2D_idx2;
      corr_data2.image = &image2;
      corr_data2.camera = &camera2;
      corr_data2.point2D = &point2D2;

      // 根据2D点是否已有关联的3D点，分三种情况处理：

      // 情况1：第一个点有3D点，第二个点没有
      // 尝试将第二个点继续到现有的3D点轨迹中
      if (point2D1.HasPoint3D() && !point2D2.HasPoint3D()) {
        const std::vector<CorrData> corrs_data1 = {corr_data1};
        num_tris += Continue(re_options, corr_data2, corrs_data1);
      } 
      // 情况2：第二个点有3D点，第一个点没有
      // 尝试将第一个点继续到现有的3D点轨迹中
      else if (!point2D1.HasPoint3D() && point2D2.HasPoint3D()) {
        const std::vector<CorrData> corrs_data2 = {corr_data2};
        num_tris += Continue(re_options, corr_data1, corrs_data2);
      } 
      // 情况3：两个点都没有3D点
      // 尝试创建新的3D点
      else if (!point2D1.HasPoint3D() && !point2D2.HasPoint3D()) {
        const std::vector<CorrData> corrs_data = {corr_data1, corr_data2};
        // 注意：这里使用原始选项而不是宽松的重三角化选项
        // 创建新点时使用更严格的阈值，避免引入低质量的3D点导致漂移
        // Do not use larger triangulation threshold as this causes
        // significant drift when creating points (options vs. re_options).
        num_tris += Create(options, corrs_data);
      }
      // 如果两个点都已有3D点，则跳过
      // 在重三角化中我们不合并已有的3D点，避免破坏现有结构
      // Else both points have a 3D point, but we do not want to
      // merge points in retriangulation.
    }
  }

  // 返回本次重三角化创建的3D点总数
  return num_tris;
}

void IncrementalTriangulator::AddModifiedPoint3D(const point3D_t point3D_id) {
  modified_point3D_ids_.insert(point3D_id);
}

const std::unordered_set<point3D_t>&
IncrementalTriangulator::GetModifiedPoints3D() {
  // First remove any missing 3D points from the set.
  for (auto it = modified_point3D_ids_.begin();
       it != modified_point3D_ids_.end();) {
    if (reconstruction_->ExistsPoint3D(*it)) {
      ++it;
    } else {
      modified_point3D_ids_.erase(it++);
    }
  }
  return modified_point3D_ids_;
}

void IncrementalTriangulator::ClearModifiedPoints3D() {
  modified_point3D_ids_.clear();
}

void IncrementalTriangulator::ClearCaches() {
  camera_has_bogus_params_.clear();
  merge_trials_.clear();
  found_corrs_.clear();
}

/**
 * 查找指定2D点的对应关系
 * 
 * 这个函数是三角化过程中的核心步骤，负责寻找给定图像中某个2D特征点
 * 在其他已注册图像中的对应点。支持直接匹配和传递性匹配两种模式。
 * 
 * @param options 三角化选项参数
 * @param image_id 参考图像ID
 * @param point2D_idx 参考图像中2D点的索引
 * @param transitivity 传递性搜索深度，1表示直接匹配，>1表示多跳匹配
 * @param corrs_data 输出参数，存储找到的有效对应关系数据
 * @return 已经三角化的对应点数量
 */
size_t IncrementalTriangulator::Find(const Options& options,
                                     const image_t image_id,
                                     const point2D_t point2D_idx,
                                     const size_t transitivity,
                                     std::vector<CorrData>* corrs_data) {
  
  // 声明指向对应关系列表的指针，用于统一处理不同搜索模式的结果
  const std::vector<CorrespondenceGraph::Correspondence>* found_corrs_ptr =
      nullptr;
  
  // 根据传递性参数选择不同的搜索策略
  if (transitivity == 1) {
    // 直接匹配模式：只查找直接连接的对应关系
    // 这是最常见的情况，查找与当前2D点直接匹配的其他图像中的点
    found_corrs_ptr =
        &correspondence_graph_->FindCorrespondences(image_id, point2D_idx);
  } else {
    // 传递性匹配模式：查找多跳连接的对应关系
    // 例如：A图像中的点1匹配B图像中的点2，B图像中的点2匹配C图像中的点3
    // 则A图像中的点1与C图像中的点3形成传递性对应关系
    // 这种方式可以发现更多的对应关系，提高三角化的成功率
    correspondence_graph_->FindTransitiveCorrespondences(
        image_id, point2D_idx, transitivity, &found_corrs_);
    found_corrs_ptr = &found_corrs_;  // 使用成员变量存储传递性搜索结果
  }

  // 清空输出容器并预分配空间，提高性能
  corrs_data->clear();
  corrs_data->reserve(found_corrs_ptr->size());

  // 统计已经三角化的对应点数量
  // 这个数字将用于后续的处理策略选择
  size_t num_triangulated = 0;

  // 遍历所有找到的对应关系，进行有效性检查和数据准备
  for (const auto& corr : *found_corrs_ptr) {
    
    // 获取对应点所在的图像对象
    const Image& corr_image = reconstruction_->Image(corr.image_id);
    
    // 检查对应图像是否已经注册
    // 未注册的图像没有已知的相机位姿，无法用于三角化
    if (!corr_image.IsRegistered()) {
      continue;  // 跳过未注册的图像
    }

    // 获取对应图像的相机参数
    const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());
    
    // 检查相机参数是否异常
    // 异常的相机参数（如过大的焦距、畸变系数等）会导致不可靠的三角化
    if (HasCameraBogusParams(options, corr_camera)) {
      continue;  // 跳过参数异常的相机
    }

    // 为当前对应关系创建完整的数据结构
    // CorrData包含了进行三角化所需的所有信息
    CorrData corr_data;
    corr_data.image_id = corr.image_id;           // 对应图像的ID
    corr_data.point2D_idx = corr.point2D_idx;    // 对应点在图像中的索引
    corr_data.image = &corr_image;               // 对应图像对象的指针
    corr_data.camera = &corr_camera;             // 对应相机对象的指针
    corr_data.point2D = &corr_image.Point2D(corr.point2D_idx);  // 对应2D点对象的指针

    // 将有效的对应关系添加到输出列表中
    corrs_data->push_back(corr_data);

    // 检查当前对应点是否已经关联到某个3D点
    // 已三角化的点可以用于Continue操作（将新点关联到已有轨迹）
    if (corr_data.point2D->HasPoint3D()) {
      num_triangulated += 1;  // 增加已三角化计数
    }
  }

  // 返回已经三角化的对应点数量
  // 这个信息将帮助调用者决定使用Create还是Continue策略：
  // - 如果num_triangulated == 0：所有点都未三角化，使用Create创建新3D点
  // - 如果num_triangulated > 0：部分点已三角化，先尝试Continue再Create
  return num_triangulated;
}

size_t IncrementalTriangulator::Create(
    const Options& options, const std::vector<CorrData>& corrs_data) {
  // Extract correspondences without an existing triangulated observation.
  std::vector<CorrData> create_corrs_data;
  create_corrs_data.reserve(corrs_data.size());
  for (const CorrData& corr_data : corrs_data) {
    if (!corr_data.point2D->HasPoint3D()) {
      create_corrs_data.push_back(corr_data);
    }
  }

  if (create_corrs_data.size() < 2) {
    // Need at least two observations for triangulation.
    return 0;
  } else if (options.ignore_two_view_tracks && create_corrs_data.size() == 2) {
    const CorrData& corr_data1 = create_corrs_data[0];
    if (correspondence_graph_->IsTwoViewObservation(corr_data1.image_id,
                                                    corr_data1.point2D_idx)) {
      return 0;
    }
  }

  // Setup data for triangulation estimation.
  std::vector<TriangulationEstimator::PointData> point_data;
  point_data.resize(create_corrs_data.size());
  std::vector<TriangulationEstimator::PoseData> pose_data;
  pose_data.resize(create_corrs_data.size());
  for (size_t i = 0; i < create_corrs_data.size(); ++i) {
    const CorrData& corr_data = create_corrs_data[i];
    point_data[i].point = corr_data.point2D->XY();
    point_data[i].point_normalized =
        corr_data.camera->ImageToWorld(point_data[i].point);
    pose_data[i].proj_matrix = corr_data.image->ProjectionMatrix();
    pose_data[i].proj_center = corr_data.image->ProjectionCenter();
    pose_data[i].camera = corr_data.camera;
  }

  // Setup estimation options.
  EstimateTriangulationOptions tri_options;
  tri_options.min_tri_angle = DegToRad(options.min_angle);
  tri_options.residual_type =
      TriangulationEstimator::ResidualType::ANGULAR_ERROR;
  tri_options.ransac_options.max_error =
      DegToRad(options.create_max_angle_error);
  tri_options.ransac_options.confidence = 0.9999;
  tri_options.ransac_options.min_inlier_ratio = 0.02;
  tri_options.ransac_options.max_num_trials = 10000;

  // Enforce exhaustive sampling for small track lengths.
  const size_t kExhaustiveSamplingThreshold = 15;
  if (point_data.size() <= kExhaustiveSamplingThreshold) {
    tri_options.ransac_options.min_num_trials = NChooseK(point_data.size(), 2);
  }

  // Estimate triangulation.
  Eigen::Vector3d xyz;
  std::vector<char> inlier_mask;
  if (!EstimateTriangulation(tri_options, point_data, pose_data, &inlier_mask,
                             &xyz)) {
    return 0;
  }

  // Add inliers to estimated track.
  Track track;
  track.Reserve(create_corrs_data.size());
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const CorrData& corr_data = create_corrs_data[i];
      track.AddElement(corr_data.image_id, corr_data.point2D_idx);
    }
  }

  // Add estimated point to reconstruction.
  const size_t track_length = track.Length();
  const point3D_t point3D_id =
      reconstruction_->AddPoint3D(xyz, std::move(track));
  modified_point3D_ids_.insert(point3D_id);

  const size_t kMinRecursiveTrackLength = 3;
  if (create_corrs_data.size() - track_length >= kMinRecursiveTrackLength) {
    return track_length + Create(options, create_corrs_data);
  }

  return track_length;
}

size_t IncrementalTriangulator::Continue(
    const Options& options, const CorrData& ref_corr_data,
    const std::vector<CorrData>& corrs_data) {
  // No need to continue, if the reference observation is triangulated.
  if (ref_corr_data.point2D->HasPoint3D()) {
    return 0;
  }

  double best_angle_error = std::numeric_limits<double>::max();
  size_t best_idx = std::numeric_limits<size_t>::max();

  for (size_t idx = 0; idx < corrs_data.size(); ++idx) {
    const CorrData& corr_data = corrs_data[idx];
    if (!corr_data.point2D->HasPoint3D()) {
      continue;
    }

    const Point3D& point3D =
        reconstruction_->Point3D(corr_data.point2D->Point3DId());

    const double angle_error = CalculateAngularError(
        ref_corr_data.point2D->XY(), point3D.XYZ(), ref_corr_data.image->Qvec(),
        ref_corr_data.image->Tvec(), *ref_corr_data.camera);
    if (angle_error < best_angle_error) {
      best_angle_error = angle_error;
      best_idx = idx;
    }
  }

  const double max_angle_error = DegToRad(options.continue_max_angle_error);
  if (best_angle_error <= max_angle_error &&
      best_idx != std::numeric_limits<size_t>::max()) {
    const CorrData& corr_data = corrs_data[best_idx];
    const TrackElement track_el(ref_corr_data.image_id,
                                ref_corr_data.point2D_idx);
    reconstruction_->AddObservation(corr_data.point2D->Point3DId(), track_el);
    modified_point3D_ids_.insert(corr_data.point2D->Point3DId());
    return 1;
  }

  return 0;
}

size_t IncrementalTriangulator::Merge(const Options& options,
                                      const point3D_t point3D_id) {
  if (!reconstruction_->ExistsPoint3D(point3D_id)) {
    return 0;
  }

  const double max_squared_reproj_error =
      options.merge_max_reproj_error * options.merge_max_reproj_error;

  const auto& point3D = reconstruction_->Point3D(point3D_id);

  for (const auto& track_el : point3D.Track().Elements()) {
    const std::vector<CorrespondenceGraph::Correspondence>& corrs =
        correspondence_graph_->FindCorrespondences(track_el.image_id,
                                                   track_el.point2D_idx);

    for (const auto corr : corrs) {
      const auto& image = reconstruction_->Image(corr.image_id);
      if (!image.IsRegistered()) {
        continue;
      }

      const Point2D& corr_point2D = image.Point2D(corr.point2D_idx);
      if (!corr_point2D.HasPoint3D() ||
          corr_point2D.Point3DId() == point3D_id ||
          merge_trials_[point3D_id].count(corr_point2D.Point3DId()) > 0) {
        continue;
      }

      // Try to merge the two 3D points.

      const Point3D& corr_point3D =
          reconstruction_->Point3D(corr_point2D.Point3DId());

      merge_trials_[point3D_id].insert(corr_point2D.Point3DId());
      merge_trials_[corr_point2D.Point3DId()].insert(point3D_id);

      // Weighted average of point locations, depending on track length.
      const Eigen::Vector3d merged_xyz =
          (point3D.Track().Length() * point3D.XYZ() +
           corr_point3D.Track().Length() * corr_point3D.XYZ()) /
          (point3D.Track().Length() + corr_point3D.Track().Length());

      // Count number of inlier track elements of the merged track.
      bool merge_success = true;
      for (const Track* track : {&point3D.Track(), &corr_point3D.Track()}) {
        for (const auto test_track_el : track->Elements()) {
          const Image& test_image =
              reconstruction_->Image(test_track_el.image_id);
          const Camera& test_camera =
              reconstruction_->Camera(test_image.CameraId());
          const Point2D& test_point2D =
              test_image.Point2D(test_track_el.point2D_idx);
          if (CalculateSquaredReprojectionError(
                  test_point2D.XY(), merged_xyz, test_image.Qvec(),
                  test_image.Tvec(), test_camera) > max_squared_reproj_error) {
            merge_success = false;
            break;
          }
        }
        if (!merge_success) {
          break;
        }
      }

      // Only accept merge if all track elements are inliers.
      if (merge_success) {
        const size_t num_merged =
            point3D.Track().Length() + corr_point3D.Track().Length();

        const point3D_t merged_point3D_id = reconstruction_->MergePoints3D(
            point3D_id, corr_point2D.Point3DId());

        modified_point3D_ids_.erase(point3D_id);
        modified_point3D_ids_.erase(corr_point2D.Point3DId());
        modified_point3D_ids_.insert(merged_point3D_id);

        // Merge merged 3D point and return, as the original points are deleted.
        const size_t num_merged_recursive = Merge(options, merged_point3D_id);
        if (num_merged_recursive > 0) {
          return num_merged_recursive;
        } else {
          return num_merged;
        }
      }
    }
  }

  return 0;
}

size_t IncrementalTriangulator::Complete(const Options& options,
                                         const point3D_t point3D_id) {
  size_t num_completed = 0;

  if (!reconstruction_->ExistsPoint3D(point3D_id)) {
    return num_completed;
  }

  const double max_squared_reproj_error =
      options.complete_max_reproj_error * options.complete_max_reproj_error;

  const Point3D& point3D = reconstruction_->Point3D(point3D_id);

  std::vector<TrackElement> queue = point3D.Track().Elements();

  const int max_transitivity = options.complete_max_transitivity;
  for (int transitivity = 0; transitivity < max_transitivity; ++transitivity) {
    if (queue.empty()) {
      break;
    }

    const auto prev_queue = queue;
    queue.clear();

    for (const TrackElement queue_elem : prev_queue) {
      const std::vector<CorrespondenceGraph::Correspondence>& corrs =
          correspondence_graph_->FindCorrespondences(queue_elem.image_id,
                                                     queue_elem.point2D_idx);

      for (const auto corr : corrs) {
        const Image& image = reconstruction_->Image(corr.image_id);
        if (!image.IsRegistered()) {
          continue;
        }

        const Point2D& point2D = image.Point2D(corr.point2D_idx);
        if (point2D.HasPoint3D()) {
          continue;
        }

        const Camera& camera = reconstruction_->Camera(image.CameraId());
        if (HasCameraBogusParams(options, camera)) {
          continue;
        }

        if (CalculateSquaredReprojectionError(
                point2D.XY(), point3D.XYZ(), image.Qvec(), image.Tvec(),
                camera) > max_squared_reproj_error) {
          continue;
        }

        // Success, add observation to point track.
        const TrackElement track_el(corr.image_id, corr.point2D_idx);
        reconstruction_->AddObservation(point3D_id, track_el);
        modified_point3D_ids_.insert(point3D_id);

        // Recursively complete track for this new correspondence.
        if (transitivity < max_transitivity - 1) {
          queue.emplace_back(corr.image_id, corr.point2D_idx);
        }

        num_completed += 1;
      }
    }
  }

  return num_completed;
}

bool IncrementalTriangulator::HasCameraBogusParams(const Options& options,
                                                   const Camera& camera) {
  const auto it = camera_has_bogus_params_.find(camera.CameraId());
  if (it == camera_has_bogus_params_.end()) {
    const bool has_bogus_params = camera.HasBogusParams(
        options.min_focal_length_ratio, options.max_focal_length_ratio,
        options.max_extra_param);
    camera_has_bogus_params_.emplace(camera.CameraId(), has_bogus_params);
    return has_bogus_params;
  } else {
    return it->second;
  }
}

}  // namespace colmap
