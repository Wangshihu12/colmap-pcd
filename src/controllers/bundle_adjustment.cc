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

#include "controllers/bundle_adjustment.h"

#include <ceres/ceres.h>

#include "optim/bundle_adjustment.h"
#include "util/misc.h"

namespace colmap {
namespace {

// Callback functor called after each bundle adjustment iteration.
class BundleAdjustmentIterationCallback : public ceres::IterationCallback {
 public:
  explicit BundleAdjustmentIterationCallback(Thread* thread)
      : thread_(thread) {}

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    CHECK_NOTNULL(thread_);
    thread_->BlockIfPaused();
    if (thread_->IsStopped()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    } else {
      return ceres::SOLVER_CONTINUE;
    }
  }

 private:
  Thread* thread_;
};

}  // namespace

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options, Reconstruction* reconstruction)
    : options_(options), reconstruction_(reconstruction) {}
// bundle adjustment批优化

/**
 * 全局束调整控制器的主运行函数
 * 
 * 该函数负责执行重建过程中的全局束调整优化，包括标准优化和集成激光雷达约束的优化。
 * 全局束调整同时优化所有相机位姿和三维点坐标，以最小化重投影误差和可能的激光雷达约束误差。
 */
void BundleAdjustmentController::Run() {
  // 确保重建对象已经初始化
  CHECK_NOTNULL(reconstruction_);

  // 打印全局束调整的标题
  PrintHeading1("Global bundle adjustment");

  // 获取所有已注册图像的ID列表
  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  // 检查已注册图像数量是否足够进行束调整（至少需要两张图像）
  if (reg_image_ids.size() < 2) {
    std::cout << "ERROR: Need at least two views." << std::endl;
    return;
  }

  // 过滤具有负深度的观测点，这些点通常是错误的三角化结果
  // 负深度意味着点在相机后方，这在物理上是不可能的
  reconstruction_->FilterObservationsWithNegativeDepth();

  // 复制全局束调整选项并启用进度输出到标准输出
  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
  ba_options.solver_options.minimizer_progress_to_stdout = true;
  
  // 创建迭代回调对象并添加到求解器选项中
  // 该回调可用于监控优化过程、处理用户交互或可视化
  BundleAdjustmentIterationCallback iteration_callback(this);
  ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // 配置束调整优化问题
  BundleAdjustmentConfig ba_config;

  // 将所有已注册图像添加到优化配置中
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // 如果启用了激光雷达约束，执行相关设置
  if (ba_options.if_add_lidar_constraint){

    // 清除之前可能存在的激光雷达点
    ClearLidarPoints();

    // 获取增量式重建器选项
    IncrementalMapperOptions mapper_options = *options_.mapper;

    // 获取激光雷达点云路径并加载点云数据
    std::string path = mapper_options.lidar_pointcloud_path;
    LoadPointcloud(path, mapper_options.PcdProjector());

    // 获取重建中所有3D点的ID集合
    std::unordered_set<point3D_t> reg_point3D_ids = reconstruction_->Point3DIds();

    // 遍历每个3D点，为其建立与激光雷达点云的对应关系
    for (point3D_t point3d_id : reg_point3D_ids) {

      // 将3D点添加为优化变量
      ba_config.AddVariablePoint(point3d_id);
      
      // 获取3D点的坐标
      Point3D& point3D = reconstruction_->Point3D(point3d_id);
      Eigen::Vector3d pt_xyz = point3D.XYZ();

      // 使用KD树在激光雷达点云中搜索最近邻点
      // Vector6d存储的是[x,y,z,nx,ny,nz]，即点的坐标和法向量
      Eigen::Vector6d lidar_pt;
      if (lidar_pointcloud_process_->SearchNearestNeiborByKdtree(pt_xyz,lidar_pt)){
        // 提取法向量和点坐标
        Eigen::Vector3d norm = lidar_pt.block(3,0,3,1);
        Eigen::Vector3d l_pt = lidar_pt.block(0,0,3,1);

        // 计算平面方程的d参数（ax+by+cz+d=0形式中的d）
        double d = 0 - l_pt.dot(norm);
        Eigen::Vector4d plane;
        plane << norm(0),norm(1),norm(2),d;

        // 创建激光雷达点对象，包含点坐标和所在平面信息
        LidarPoint lidar_point(l_pt,plane);

        // 计算3D点到激光雷达平面的距离和到点的距离
        const double dist2plane = lidar_point.ComputeDist(pt_xyz);
        const double dist2point = lidar_point.ComputePointToPointDist(pt_xyz);
        
        // 如果距离过大，说明对应关系不可靠，跳过该点
        if (dist2plane > 1 || dist2point > 2) continue;

        // 根据法向量特征判断点的类型（地面点或普通点）
        // 当法向量y分量远大于x和z分量时，认为是地面点
        if (std::abs(norm(1)/norm(0))>10 && std::abs(norm(1)/norm(2))>10) {
          // 设置为地面点类型（用于地面特殊约束）
          lidar_point.SetType(LidarPointType::IcpGround);
          Eigen::Vector3ub color;
          color << 255,255,0;
          lidar_point.SetColor(color);
        } else {
          // 设置为普通ICP点类型
          lidar_point.SetType(LidarPointType::Icp);
          Eigen::Vector3ub color;
          color << 0,0,255;
          lidar_point.SetColor(color);
        }

        // 将3D点与激光雷达点的对应关系添加到束调整配置中
        ba_config.AddLidarPoint(point3d_id,lidar_point);

        // 在全局重建中也添加该对应关系（用于可视化或后续处理）
        reconstruction_ -> AddLidarPointInGlobal(point3d_id,lidar_point);

      }
    }
  } else {

    // 如果不使用激光雷达约束，采用传统SfM的约束条件
    // 固定第一个相机的所有参数和第二个相机的平移x分量
    // 这是为了消除尺度和坐标系的不确定性
    ba_config.SetConstantPose(reg_image_ids[0]);
    ba_config.SetConstantTvec(reg_image_ids[1], {0});
  }

  // 创建束调整器并设置为全局优化模式
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  const BundleAdjuster::OptimazePhrase phrase = BundleAdjuster::OptimazePhrase::WholeMap;
  bundle_adjuster.SetOptimazePhrase(phrase);

  // 执行束调整优化
  bundle_adjuster.Solve(reconstruction_);

  GetTimer().PrintMinutes();
}

void BundleAdjustmentController::LoadPointcloud(std::string& pointcloud_path, 
                                       const lidar::PcdProjectionOptions& pp_options){
  lidar_pointcloud_process_.reset(new lidar::PointCloudProcess(pointcloud_path));
  if (!lidar_pointcloud_process_->Initialize(pp_options)){
    std::cout<< "Point cloud initialize has error"<<std::endl;
  }
}

void BundleAdjustmentController::ClearLidarPoints(){
  reconstruction_->ClearLidarPoints();
  reconstruction_->ClearLidarPointsInGlobal();
}

}  // namespace colmap
