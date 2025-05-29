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

#include "base/database_cache.h"

#include <unordered_set>

#include "feature/utils.h"
#include "util/string.h"
#include "util/timer.h"

namespace colmap {

DatabaseCache::DatabaseCache() {}

void DatabaseCache::AddCamera(class Camera camera) {
  const camera_t camera_id = camera.CameraId();
  CHECK(!ExistsCamera(camera_id));
  cameras_.emplace(camera_id, std::move(camera));
}

void DatabaseCache::AddImage(class Image image) {
  const image_t image_id = image.ImageId();
  CHECK(!ExistsImage(image_id));
  correspondence_graph_.AddImage(image_id, image.NumPoints2D());
  images_.emplace(image_id, std::move(image));
}

/**
 * [功能描述]：从数据库加载相机、图像和匹配信息，并构建对应关系图
 * 此函数是SfM系统初始化的关键部分，将数据库中的所有必要信息加载到内存中
 * @param [database]：包含所有图像、相机和匹配信息的数据库
 * @param [min_num_matches]：要考虑的两视图几何体的最小内点匹配数
 * @param [ignore_watermarks]：是否忽略水印标记的匹配
 * @param [image_names]：要加载的特定图像名称集合，为空则加载所有图像
 */
void DatabaseCache::Load(const Database& database, const size_t min_num_matches,
                         const bool ignore_watermarks,
                         const std::unordered_set<std::string>& image_names) {
  //////////////////////////////////////////////////////////////////////////////
  // Load cameras
  //////////////////////////////////////////////////////////////////////////////

  Timer timer; // 计时器，用于记录各个加载步骤的耗时

  timer.Start();
  std::cout << "Loading cameras..." << std::flush; // 打印加载进度

  //导入相机模型
  {
    std::vector<class Camera> cameras = database.ReadAllCameras();
    cameras_.reserve(cameras.size()); // 预分配内存，提高性能
    for (auto& camera : cameras) {
      const camera_t camera_id = camera.CameraId();
      cameras_.emplace(camera_id, std::move(camera)); // 移动语义，避免复制
    }
  }

  // 打印加载结果和耗时
  std::cout << StringPrintf(" %d in %.3fs", cameras_.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Load matches
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading matches..." << std::flush;

  // 读取所有两视图几何信息
  std::vector<image_pair_t> image_pair_ids; // 图像对ID
  std::vector<TwoViewGeometry> two_view_geometries; // 两视图几何数据
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

  std::cout << StringPrintf(" %d in %.3fs", image_pair_ids.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  // 定义一个lambda函数，用于检查两视图几何是否满足使用条件
  // 条件1：内点匹配数量不少于最小阈值
  // 条件2：如果忽略水印，则跳过水印标记的几何体
  auto UseInlierMatchesCheck = [min_num_matches, ignore_watermarks](
                                   const TwoViewGeometry& two_view_geometry) {
    return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >=
               min_num_matches &&
           (!ignore_watermarks ||
            two_view_geometry.config != TwoViewGeometry::WATERMARK);
  };

  //////////////////////////////////////////////////////////////////////////////
  // Load images 导入图像
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading images..." << std::flush;

  // 用于存储需要处理的图像ID
  std::unordered_set<image_t> image_ids;

  {
    // 读取所有图像数据
    std::vector<class Image> images = database.ReadAllImages();
    const size_t num_images = images.size();

    // 确定应该加载哪些图像的数据
    if (image_names.empty()) {
      // 如果没有指定特定图像名称，则加载所有图像
      for (const auto& image : images) {
        image_ids.insert(image.ImageId());
      }
    } else {
      // 否则，只加载指定名称的图像
      for (const auto& image : images) {
        if (image_names.count(image.Name()) > 0) {
          image_ids.insert(image.ImageId());
        }
      }
    }

    // 收集所有在对应关系图中连接的图像
    // 只有那些有足够匹配的图像才会被纳入重建过程
    std::unordered_set<image_t> connected_image_ids;
    connected_image_ids.reserve(image_ids.size());
    for (size_t i = 0; i < image_pair_ids.size(); ++i) {
      if (UseInlierMatchesCheck(two_view_geometries[i])) {
        // 解析图像对ID获取两个图像的ID
        image_t image_id1;
        image_t image_id2;
        Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
        // 如果两个图像都在待处理列表中，则将它们标记为已连接
        if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
          connected_image_ids.insert(image_id1);
          connected_image_ids.insert(image_id2);
        }
      }
    }

    // 加载有对应关系的图像并丢弃没有对应关系的图像
    // 因为没有对应关系的图像对SfM过程无用
    images_.reserve(connected_image_ids.size());
    for (auto& image : images) {
      const image_t image_id = image.ImageId();
      // 只处理在待处理列表且有足够连接的图像
      if (image_ids.count(image_id) > 0 &&
          connected_image_ids.count(image_id) > 0) {
        // 添加图像到缓存
        images_.emplace(image_id, std::move(image));
        // 读取该图像的特征点
        const FeatureKeypoints keypoints = database.ReadKeypoints(image_id);
        // 将特征点转换为点向量
        const std::vector<Eigen::Vector2d> points =
            FeatureKeypointsToPointsVector(keypoints);
        // 设置图像的特征点
        images_[image_id].SetPoints2D(points);
      }
    }

    // 打印加载结果和耗时，以及连接的图像数量
    std::cout << StringPrintf(" %d in %.3fs (connected %d)", num_images,
                              timer.ElapsedSeconds(),
                              connected_image_ids.size())
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build correspondence graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Building correspondence graph..." << std::flush;

  // 为每个图像在对应关系图中添加节点
  for (const auto& image : images_) {
    correspondence_graph_.AddImage(image.first, image.second.NumPoints2D());
  }

  // 添加图像间的对应关系
  size_t num_ignored_image_pairs = 0; // 记录被忽略的图像对数量
  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    // 检查该图像对是否满足使用条件
    if (UseInlierMatchesCheck(two_view_geometries[i])) {
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
      // 确保两个图像都在待处理列表中
      if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
        // 向对应关系图添加这对图像之间的对应点
        correspondence_graph_.AddCorrespondences(
            image_id1, image_id2, two_view_geometries[i].inlier_matches);
      } else {
        num_ignored_image_pairs += 1; // 记录被忽略的图像对
      }
    } else {
      num_ignored_image_pairs += 1; // 记录被忽略的图像对
    }
  }

  // 完成对应关系图的构建，计算各种统计信息
  correspondence_graph_.Finalize();

  // 为每个图像设置观测数和对应点数
  for (auto& image : images_) {
    // 设置观测数量（该图像的特征点被其他图像观测到的次数）
    image.second.SetNumObservations(
        correspondence_graph_.NumObservationsForImage(image.first));
    // 设置对应点数量（该图像的特征点与其他图像特征点的对应关系数）
    image.second.SetNumCorrespondences(
        correspondence_graph_.NumCorrespondencesForImage(image.first));
  }

  // 打印构建结果、耗时和被忽略的图像对数量
  std::cout << StringPrintf(" in %.3fs (ignored %d)", timer.ElapsedSeconds(),
                            num_ignored_image_pairs)
            << std::endl;
}

const class Image* DatabaseCache::FindImageWithName(
    const std::string& name) const {
  for (const auto& image : images_) {
    if (image.second.Name() == name) {
      return &image.second;
    }
  }
  return nullptr;
}

}  // namespace colmap
