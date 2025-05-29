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

#include "optim/bundle_adjustment.h"

#include <iomanip>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/projection.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction* BundleAdjustmentOptions::CreateLossFunction() const {
  ceres::LossFunction* loss_function = nullptr;
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      loss_function = new ceres::TrivialLoss();
      break;
    case LossFunctionType::SOFT_L1:
      loss_function = new ceres::SoftLOneLoss(loss_function_scale);
      break;
    case LossFunctionType::CAUCHY:
      loss_function = new ceres::CauchyLoss(loss_function_scale);
      break;
  }
  CHECK_NOTNULL(loss_function);
  return loss_function;
}

bool BundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentConfig::BundleAdjustmentConfig() {
  lidar_searched_image_ids_.clear();
  lidar_maps_.clear();
}

// image_ids_是一个包含image_id的set
size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCameras() const {
  return constant_camera_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoses() const {
  return constant_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantTvecs() const {
  return constant_tvecs_.size();
}

size_t BundleAdjustmentConfig::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumResiduals(
    const Reconstruction& reconstruction) const {
  // Count the number of observations for all added images.
  size_t num_observations = 0;

  for (const image_t image_id : image_ids_) {
    num_observations += reconstruction.Image(image_id).NumPoints3D();
  }

  // Count the number of observations for all added 3D points that are not
  // already added as part of the images above.

  auto NumObservationsForPoint = [this,
                                  &reconstruction](const point3D_t point3D_id) {
    size_t num_observations_for_point = 0;
    const auto& point3D = reconstruction.Point3D(point3D_id);
    for (const auto& track_el : point3D.Track().Elements()) {
      if (image_ids_.count(track_el.image_id) == 0) {
        num_observations_for_point += 1;
      }
    }
    return num_observations_for_point;
  };

  for (const auto point3D_id : variable_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }
  for (const auto point3D_id : constant_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }

  return 2 * num_observations;
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

void BundleAdjustmentConfig::AddLidarPoint(const point3D_t& point3D_id, const class LidarPoint& lidar_point) {
  lidar_maps_.insert({point3D_id,lidar_point});
}

void BundleAdjustmentConfig::AddPointcloud(std::shared_ptr<lidar::PointCloudProcess> ptr) {
  point_cloud_process_ = ptr;
}

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void BundleAdjustmentConfig::SetConstantCamera(const camera_t camera_id) {
  constant_camera_ids_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamera(const camera_t camera_id) {
  constant_camera_ids_.erase(camera_id);
}

bool BundleAdjustmentConfig::IsConstantCamera(const camera_t camera_id) const {
  return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
}

void BundleAdjustmentConfig::SetConstantPose(const image_t image_id) {
  CHECK(HasImage(image_id));
  CHECK(!HasConstantTvec(image_id));
  constant_poses_.insert(image_id);
}

void BundleAdjustmentConfig::SetVariablePose(const image_t image_id) {
  constant_poses_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantPose(const image_t image_id) const {
  return constant_poses_.find(image_id) != constant_poses_.end();
}

void BundleAdjustmentConfig::SetConstantTvec(const image_t image_id,
                                             const std::vector<int>& idxs) {
  CHECK_GT(idxs.size(), 0);
  CHECK_LE(idxs.size(), 3);
  CHECK(HasImage(image_id));
  CHECK(!HasConstantPose(image_id));
  CHECK(!VectorContainsDuplicateValues(idxs))
      << "Tvec indices must not contain duplicates";
  constant_tvecs_.emplace(image_id, idxs);
}

void BundleAdjustmentConfig::RemoveConstantTvec(const image_t image_id) {
  constant_tvecs_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantTvec(const image_t image_id) const {
  return constant_tvecs_.find(image_id) != constant_tvecs_.end();
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::Images() const {
  return image_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::VariablePoints()
    const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::ConstantPoints()
    const {
  return constant_point3D_ids_;
}

const std::vector<int>& BundleAdjustmentConfig::ConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.at(image_id);
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
  CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

/**
 * [功能描述]：将图像中的特征点投影到点云中，并计算投影尺度
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @param [point3D_id]：待处理的三维点ID
 * @param [image_id]：参考图像ID
 * @param [match_features_threshold]：特征匹配数量阈值，用于筛选有效的图像对
 */
void BundleAdjustmentConfig::Project2Image(Reconstruction* reconstruction,const point3D_t& point3D_id, const image_t& image_id, const int& match_features_threshold) {
  // 获取指定ID的三维点
  Point3D& point3D = reconstruction->Point3D(point3D_id);
  
  // 获取该三维点的观测轨迹(track)，包含了所有观察到该点的图像及其二维坐标
  const auto& track_els = point3D.Track().Elements();

  // 遍历该三维点的所有观测图像
  for (auto& track_el : track_els){
    // 获取当前观测图像的ID
    image_t image_id2 = track_el.image_id;
    // 如果当前图像与参考图像不同
    if (image_id2 != image_id) {
      // 检查两图像是否构成有效的图像对
      if (reconstruction->ExistsImagePair(image_id,image_id2)){
        // 获取两图像间的共同特征点数量
        size_t corrs = reconstruction->ImagePair(image_id, image_id2).num_total_corrs;
        // 如果共同特征点数量不足阈值，跳过该图像
        if (corrs <= match_features_threshold) {
          continue;
        }
      }
    }
    
    // 检查当前图像是否已经被处理过（投影到点云中）
    auto ptr = lidar_searched_image_ids_.find(track_el.image_id);
    if (ptr == lidar_searched_image_ids_.end()){
      // 如果图像未被处理过，则进行处理
      // 获取图像和相机信息
      Image& image = reconstruction->Image(track_el.image_id);
      Camera& camera = reconstruction->Camera(image.CameraId());

      // 创建映射表存储投影结果
      std::map<point3D_t,Eigen::Matrix<double,6,1>> map;
      // 调用点云处理器的投影模块，将图像中的特征点投影到点云空间
      point_cloud_process_ -> pcd_proj_ -> SetNewImage(image,camera,map);
      // 记录该图像已被处理，并存储投影结果
      lidar_searched_image_ids_.insert({track_el.image_id,map});
    }
  }
}

/**
 * [功能描述]：将SfM三维点与激光雷达点云进行匹配，并选择最优匹配点
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景数据
 * @param [point3D_id]：待匹配的三维点ID
 */
void BundleAdjustmentConfig::MatchVariablePoint2LidarPoint(Reconstruction* reconstruction,const point3D_t point3D_id){
  // 获取指定ID的三维点及其坐标
  Point3D& point3D = reconstruction->Point3D(point3D_id);
  Eigen::Vector3d pt_xyz = point3D.XYZ();
  // 初始化最小角度为一个很大的值(360度)，用于寻找最佳匹配点
  double angle = 360;
  // 用于存储最佳匹配的激光雷达点(前3维为位置，后3维为法向量)
  Eigen::Vector6d lidar_pt;

  // 获取该三维点的观测轨迹(所有观察到该点的图像)
  const auto& track_els = point3D.Track().Elements(); 
  // 遍历每个观测图像
  for (auto& track_el : track_els){
    // 检查图像是否已被处理过(是否已投影到点云中)
    auto iter = lidar_searched_image_ids_.find(track_el.image_id);
    if (iter == lidar_searched_image_ids_.end()) continue; // 若未处理过则跳过

    // 在处理结果中查找当前三维点对应的激光雷达点
    auto lidar_pt_ptr = iter->second.find(point3D_id);
    if (lidar_pt_ptr != iter->second.end()){
      // 获取激光雷达点信息
      Eigen::Matrix<double, 6, 1> lidar_pt_temp = lidar_pt_ptr->second;
      // 提取法向量(后3维)
      Eigen::Vector3d norm = lidar_pt_temp.block(3,0,3,1);
      // 计算三维点到激光雷达点的向量
      Eigen::Vector3d vec = pt_xyz - lidar_pt_temp.block(0,0,3,1);
      // 计算向量与法向量的夹角余弦值的绝对值
      double angle_temp = std::abs(vec.dot(norm)/(norm.norm()*vec.norm()));
      // 如果角度更小，更新最佳匹配点
      if (angle_temp < angle){
        angle = angle_temp;
        lidar_pt = lidar_pt_temp;
      }
    }
  }

  // 如果找到了合适的匹配点(角度不为初始值)
  if (angle != 360){
    // 提取最佳匹配点的法向量和位置
    Eigen::Vector3d norm = lidar_pt.block(3,0,3,1);
    Eigen::Vector3d l_pt = lidar_pt.block(0,0,3,1);
    // 计算平面方程的d参数(平面方程ax+by+cz+d=0中的d)
    double d = 0 - l_pt.dot(norm);
    // 构建平面参数向量[a,b,c,d]
    Eigen::Vector4d plane;
    plane << norm(0),norm(1),norm(2),d;

    // 创建激光雷达点对象，类型为投影点(Proj)
    LidarPoint lidar_point(LidarPointType::Proj,l_pt,plane);
    // 计算并设置三维点到平面的距离
    lidar_point.SetDist( lidar_point.ComputeDist(pt_xyz));
    // 计算并设置三维点与平面法向量的夹角
    lidar_point.SetAngle( lidar_point.ComputeAngle(pt_xyz));
    // 设置点的颜色为红色(用于可视化)
    Eigen::Vector3ub color;
    color << 255,0,0;
    lidar_point.SetColor(color);

    // 将匹配结果添加到当前配置和重建对象中
    AddLidarPoint(point3D_id,lidar_point);
    reconstruction -> AddLidarPoint(point3D_id,lidar_point);
  }
}

/**
 * [功能描述]：利用KD树搜索与SfM三维点最接近的激光雷达点
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @param [point3D_id]：待匹配的三维点ID
 * @param [max_search_range]：最大搜索范围，用于筛选有效匹配
 */
void BundleAdjustmentConfig::MatchClosestLidarPoint(Reconstruction* reconstruction, 
                                                    const point3D_t& point3D_id, 
                                                    double& max_search_range){
  // 获取指定ID的三维点及其坐标
  Point3D& point3D = reconstruction->Point3D(point3D_id);
  Eigen::Vector3d pt_xyz = point3D.XYZ();
  // 用于存储匹配的激光雷达点(前3维为位置，后3维为法向量)
  Eigen::Vector6d lidar_pt;

  // 使用KD树搜索最近邻点，返回结果保存在lidar_pt中
  if (point_cloud_process_->SearchNearestNeiborByKdtree(pt_xyz,lidar_pt)){
    // 提取法向量(后3维)和位置(前3维)
    Eigen::Vector3d norm = lidar_pt.block(3,0,3,1);
    Eigen::Vector3d l_pt = lidar_pt.block(0,0,3,1);
    // 计算平面方程的d参数(平面方程ax+by+cz+d=0中的d)
    double d = 0 - l_pt.dot(norm);
    // 构建平面参数向量[a,b,c,d]
    Eigen::Vector4d plane;
    plane << norm(0),norm(1),norm(2),d;

    // 创建激光雷达点对象
    LidarPoint lidar_point(l_pt,plane);
    // 判断点的类型：如果法向量的y分量远大于x和z分量，认为是地面点
    if (std::abs(norm(1)/norm(0))>10 && std::abs(norm(1)/norm(2))>10) {
      // 设置为地面点类型
      lidar_point.SetType(LidarPointType::IcpGround);
      // 设置黄色(用于可视化)
      Eigen::Vector3ub color;
      color << 255,255,0;
      lidar_point.SetColor(color);
    } else {
      // 设置为普通ICP点类型
      lidar_point.SetType(LidarPointType::Icp);
      // 设置绿色(用于可视化)
      Eigen::Vector3ub color;
      color << 0,255,0;
      lidar_point.SetColor(color);
    }

    // 计算三维点到激光雷达点的距离
    double dist = lidar_point.ComputePointToPointDist(pt_xyz);
    // 如果距离超过最大搜索范围，则放弃该匹配
    if (dist > max_search_range) return;

    // 设置距离和角度属性
    lidar_point.SetDist(dist);
    lidar_point.SetAngle( lidar_point.ComputeAngle(pt_xyz));

    // 将匹配结果添加到当前配置和重建对象中
    AddLidarPoint(point3D_id,lidar_point);
    reconstruction -> AddLidarPoint(point3D_id,lidar_point);
  }
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
  CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
}

bool BundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
}

void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjuster
////////////////////////////////////////////////////////////////////////////////

BundleAdjuster::BundleAdjuster(const BundleAdjustmentOptions& options,
                               const BundleAdjustmentConfig& config)
    : options_(options), config_(config) {
  CHECK(options_.Check());
}

void BundleAdjuster::SetOptimazePhrase(const OptimazePhrase& phrase) {
  optimize_phrase_ = phrase;
}

/**
 * [功能描述]：执行光束法平差(Bundle Adjustment)优化过程
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @return [bool]：优化是否成功执行
 */
bool BundleAdjuster::Solve(Reconstruction* reconstruction) {
  // 确保reconstruction指针非空
  CHECK_NOTNULL(reconstruction);
  // 确保BundleAdjuster没有被重复使用
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  // 创建一个新的Ceres优化问题
  problem_ = std::make_unique<ceres::Problem>();

  // 创建损失函数，用于减轻异常值的影响
  ceres::LossFunction* loss_function = options_.CreateLossFunction();

  // 根据优化设置和阶段选择不同的问题构建方式
  if(options_.if_add_lidar_constraint && optimize_phrase_ == OptimazePhrase::Local) {
    // 局部优化模式 + 激光雷达约束
    SetUpLocalByLidar(reconstruction, loss_function);
  } else if (options_.if_add_lidar_constraint && optimize_phrase_ == OptimazePhrase::Global) {
    // 全局优化模式 + 激光雷达约束
    SetUpGlobalByLidar(reconstruction, loss_function);
  } else if (options_.if_add_lidar_constraint && optimize_phrase_ == OptimazePhrase::WholeMap) {
    // 整体地图优化模式 + 激光雷达约束
    SetUpAdjustWholeMapByLidar(reconstruction, loss_function);
  } else if (!options_.if_add_lidar_constraint) {
    // 传统优化模式，不使用激光雷达约束
    SetUp(reconstruction, loss_function);
  } else {
    // 无效的优化类型配置
    std::cout<<"The correct optimization type is missing, error occurred."<<std::endl;
  }
  
  // 检查优化问题是否包含有效的残差项，如果没有则终止优化
  if (problem_->NumResiduals() == 0) {
    return false;
  }

  // 获取求解器配置选项
  ceres::Solver::Options solver_options = options_.solver_options;
  // 检查是否可以使用稀疏线性代数库
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;

  // 根据图像数量经验性地选择适合的线性求解器类型
  const size_t kMaxNumImagesDirectDenseSolver = 50; // 稠密求解器最大图像数
  const size_t kMaxNumImagesDirectSparseSolver = 1000; // 稀疏直接求解器最大图像数
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    // 图像较少时使用稠密Schur补求解器
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
    // 图像适中且有稀疏库时使用稀疏Schur补求解器
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // 图像较多时使用迭代式Schur补求解器
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  // 根据残差项数量决定是否使用多线程
  if (problem_->NumResiduals() <
      options_.min_num_residuals_for_multi_threading) {
    // 残差项较少时使用单线程
    solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
  } else {
    // 残差项较多时使用多线程，获取有效线程数
    solver_options.num_threads =
        GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  }

  // 验证求解器选项是否有效
  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  // 执行优化求解，结果保存在summary_中
  ceres::Solve(solver_options, problem_.get(), &summary_);

  // 如果设置了进度输出，额外打印一个换行
  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  // 如果设置了打印优化结果摘要
  if (options_.print_summary) {
    PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  // 清理和结束优化过程
  TearDown(reconstruction);

  return true;
}

const ceres::Solver::Summary& BundleAdjuster::Summary() const {
  return summary_;
}

/**
 * [功能描述]：设置传统光束法平差问题（不包含激光雷达约束）
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @param [loss_function]：Ceres优化器的损失函数，用于减轻异常值影响
 */
void BundleAdjuster::SetUp(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
  // 警告：AddPointsToProblem假设AddImageToProblem已被先调用
  // 不要改变指令的顺序！

  // 第一步：处理所有图像，为每个图像添加重投影误差项
  // 遍历配置中所有需要优化的图像
  for (const image_t image_id : config_.Images()) {
    // 将图像添加到优化问题中，建立相机位姿、内参与三维点之间的约束关系
    AddImageToProblem(image_id, reconstruction, loss_function);
  }

  // 第二步：处理所有可变的三维点（参与优化的点）
  for (const auto point3D_id : config_.VariablePoints()) {
    // 将可变三维点添加到优化问题中
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  // 第三步：处理所有固定的三维点（不参与优化，但提供约束的点）
  for (const auto point3D_id : config_.ConstantPoints()) {
    // 将固定三维点添加到优化问题中
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  // 第四步：设置相机参数的优化方式（哪些参数固定，哪些参数可变）
  ParameterizeCameras(reconstruction);

  // 第五步：设置三维点的优化方式（哪些点固定，哪些点可变）
  ParameterizePoints(reconstruction);
}

/**
 * [功能描述]：设置局部光束法平差问题（包含激光雷达约束）
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @param [loss_function]：Ceres优化器的损失函数，用于减轻异常值影响
 */
void BundleAdjuster::SetUpLocalByLidar(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
  // 警告：AddPointsToProblem假设AddImageToProblem已被先调用
  // 不要改变指令的顺序！

  // 第一步：处理所有图像，为每个图像添加重投影误差项
  for (const image_t image_id : config_.Images()) {
    // 将图像添加到优化问题中，建立相机位姿、内参与三维点之间的约束关系
    AddImageToProblem(image_id, reconstruction, loss_function);
  }

  // 第二步：处理所有可变的三维点（参与优化的点）
  for (const auto point3D_id : config_.VariablePoints()) {
    // 将可变三维点添加到优化问题中
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  // 第三步（与传统BA不同）：添加激光雷达约束
  // 遍历所有已匹配的SfM点-激光雷达点对
  for (auto iter = config_.lidar_maps_.begin(); iter != config_.lidar_maps_.end(); iter++){
    // 获取对应的三维点ID
    const auto point3D_id = iter->first;
    // 为该三维点添加激光雷达约束项
    AddLidarToProblem(point3D_id, reconstruction, loss_function);
  }

  // 第四步：处理所有固定的三维点（不参与优化，但提供约束的点）
  for (const auto point3D_id : config_.ConstantPoints()) {
    // 将固定三维点添加到优化问题中
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  // 第五步：设置相机参数的优化方式（哪些参数固定，哪些参数可变）
  ParameterizeCameras(reconstruction);

  // 第六步：设置三维点的优化方式（哪些点固定，哪些点可变）
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::SetUpGlobalByLidar(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {

  for (const image_t image_id : config_.Images()) {
      AddImageInSphereToProblem(image_id, reconstruction, loss_function);
  }
  
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  for (auto iter = config_.lidar_maps_.begin(); iter != config_.lidar_maps_.end(); iter++){
    const auto point3D_id = iter->first;
    AddLidarToProblem(point3D_id, reconstruction, loss_function);
  }

  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::SetUpAdjustWholeMapByLidar(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {

  for (const image_t image_id : config_.Images()) {
      AddImageToProblem(image_id, reconstruction, loss_function);
  }
  
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  for (auto iter = config_.lidar_maps_.begin(); iter != config_.lidar_maps_.end(); iter++){
    const auto point3D_id = iter->first;
    AddLidarToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::TearDown(Reconstruction*) {
  // Nothing to do
}

/**
 * [功能描述]：将球形区域内的图像添加到优化问题中，仅处理球内的三维点
 * @param [image_id]：要添加的图像ID
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @param [loss_function]：Ceres优化器的损失函数，用于减轻异常值影响
 */
void BundleAdjuster::AddImageInSphereToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {

  // 获取指定ID的图像和相应的相机对象
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // 确保四元数表示的旋转是单位四元数（Ceres优化器的要求）
  image.NormalizeQvec();

  // 获取图像位姿参数的指针，用于后续添加到优化问题中
  double* qvec_data = image.Qvec().data(); // 旋转四元数
  double* tvec_data = image.Tvec().data(); // 平移向量
  // 获取相机内参的指针
  double* camera_params_data = camera.ParamsData();

  // 判断是否将相机位姿设为常量（不优化）
  // 如果选项中禁止优化外参，或者该图像被标记为固定位姿，则位姿为常量
  const bool constant_pose =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id);

  // 添加重投影误差项到优化问题
  size_t num_observations = 0; // 观测计数
  // 遍历图像中的所有二维特征点
  for (const Point2D& point2D : image.Points2D()) {
    // 如果特征点没有对应的三维点，则跳过
    if (!point2D.HasPoint3D()) {
      continue;
    }

    // 增加观测计数
    num_observations += 1;
    // 记录该三维点被观测的次数
    point3D_num_observations_[point2D.Point3DId()] += 1;

    // 获取对应的三维点
    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());

    // 关键差异：检查该点是否在球形区域内，如不在则跳过
    if (!point3D.IfInSphere()) {
      continue;
    }

    // 确保该三维点至少被两个图像观测到（三角测量的基本要求）
    assert(point3D.Track().Length() > 1);

    // 声明代价函数指针
    ceres::CostFunction* cost_function = nullptr;

    // 根据相机位姿是否固定，创建不同类型的代价函数
    if (constant_pose) {
      // 相机位姿固定时，使用ConstantPoseCostFunction
      // 根据相机模型类型选择对应的代价函数
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                 \
    break;

        CAMERA_MODEL_SWITCH_CASES // 宏展开处理所有相机模型类型

#undef CAMERA_MODEL_CASE
      }

      // 添加残差块：只优化三维点坐标和相机内参
      problem_->AddResidualBlock(cost_function, loss_function,
                                 point3D.XYZ().data(), camera_params_data);
    } 
    else {
      // 相机位姿可变时，使用标准的BundleAdjustmentCostFunction
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::kModelId:                                            \
    cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    break;

        CAMERA_MODEL_SWITCH_CASES // 宏展开处理所有相机模型类型

#undef CAMERA_MODEL_CASE
      }

      // 添加残差块：同时优化相机位姿、三维点坐标和相机内参
      problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                 tvec_data, point3D.XYZ().data(),
                                 camera_params_data);
    }
  }

  // 如果图像有观测到的三维点，则进行额外的参数设置
  if (num_observations > 0) {
    // 记录使用的相机ID，用于后续处理相机参数
    camera_ids_.insert(image.CameraId());

    // 如果相机位姿可变，设置位姿参数化方式
    if (!constant_pose) {
      // 设置四元数约束，确保优化过程中四元数保持单位性质
      SetQuaternionManifold(problem_.get(), qvec_data);

      // 如果部分平移分量被设为常量，则设置子集流形
      if (config_.HasConstantTvec(image_id)) {
        // 获取固定平移分量的索引
        const std::vector<int>& constant_tvec_idxs =
            config_.ConstantTvec(image_id);
        // 设置子集流形，限制特定平移分量的优化
        SetSubsetManifold(3, constant_tvec_idxs, problem_.get(), tvec_data);
      }
    }
  }
}

/**
 * [功能描述]：将图像添加到光束法平差优化问题中
 * @param [image_id]：要添加的图像ID
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @param [loss_function]：Ceres优化器的损失函数，用于减轻异常值影响
 */
void BundleAdjuster::AddImageToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  // 获取指定ID的图像和相应的相机对象
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // 确保四元数表示的旋转是单位四元数（Ceres优化器的要求）
  image.NormalizeQvec();

  // 获取图像位姿参数的指针，用于后续添加到优化问题中
  double* qvec_data = image.Qvec().data(); // 旋转四元数
  double* tvec_data = image.Tvec().data(); // 平移向量
  // 获取相机内参的指针
  double* camera_params_data = camera.ParamsData();

  // 判断是否将相机位姿设为常量（不优化）
  // 如果选项中禁止优化外参，或者该图像被标记为固定位姿，则位姿为常量
  const bool constant_pose =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id);

  // 添加重投影误差项到优化问题
  size_t num_observations = 0; // 观测计数
  // 遍历图像中的所有二维特征点
  for (const Point2D& point2D : image.Points2D()) {
    // 如果特征点没有对应的三维点，则跳过
    if (!point2D.HasPoint3D()) {
      continue;
    }

    // 增加观测计数
    num_observations += 1;
    // 记录该三维点被观测的次数，用于后续处理
    point3D_num_observations_[point2D.Point3DId()] += 1;

    // 获取对应的三维点
    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    // 确保该三维点至少被两个图像观测到（三角测量的基本要求）
    assert(point3D.Track().Length() > 1);

    // 声明代价函数指针
    ceres::CostFunction* cost_function = nullptr;

    // 根据相机位姿是否固定，创建不同类型的代价函数
    if (constant_pose) {
      // 相机位姿固定时，使用ConstantPoseCostFunction
      // 根据相机模型类型选择对应的代价函数
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                 \
    break;

        CAMERA_MODEL_SWITCH_CASES // 宏展开处理所有相机模型类型

#undef CAMERA_MODEL_CASE
      }

      // 添加残差块：只优化三维点坐标和相机内参
      problem_->AddResidualBlock(cost_function, loss_function,
                                 point3D.XYZ().data(), camera_params_data);
    } 
    else {
      // 相机位姿可变时，使用标准的BundleAdjustmentCostFunction
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::kModelId:                                            \
    cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    break;

        CAMERA_MODEL_SWITCH_CASES // 宏展开处理所有相机模型类型

#undef CAMERA_MODEL_CASE
      }

      // 添加残差块：同时优化相机位姿、三维点坐标和相机内参
      problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                 tvec_data, point3D.XYZ().data(),
                                 camera_params_data);
    }
  }

  // 如果图像有观测到的三维点，则进行额外的参数设置
  if (num_observations > 0) {
    // 记录使用的相机ID，用于后续处理相机参数
    camera_ids_.insert(image.CameraId());

    // 如果相机位姿可变，设置位姿参数化方式
    if (!constant_pose) {
      // 设置四元数约束，确保优化过程中四元数保持单位性质
      SetQuaternionManifold(problem_.get(), qvec_data);

      // 如果部分平移分量被设为常量，则设置子集流形
      if (config_.HasConstantTvec(image_id)) {
        // 获取固定平移分量的索引
        const std::vector<int>& constant_tvec_idxs =
            config_.ConstantTvec(image_id);
        // 设置子集流形，限制特定平移分量的优化
        SetSubsetManifold(3, constant_tvec_idxs, problem_.get(), tvec_data);
      }
    }
  }
}

/**
 * [功能描述]：将三维点添加到优化问题中，处理未被直接包含在优化问题中的图像对该点的观测
 * @param [point3D_id]：要添加的三维点ID
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @param [loss_function]：Ceres优化器的损失函数，用于减轻异常值影响
 */
void BundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  // 获取指定ID的三维点
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // 检查该三维点是否已完全添加到优化问题中
  // 即，该点的所有观测轨迹是否都已经通过AddImageToProblem处理过
  if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
    return; // 如果所有观测都已添加，则不需要再处理
  }

  // 遍历该三维点的所有观测轨迹元素（每个元素包含图像ID和特征点ID）
  for (const auto& track_el : point3D.Track().Elements()) {
    // 跳过已经在AddImageToProblem中添加过的图像观测
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    // 记录该三维点被处理的观测次数加1
    point3D_num_observations_[point3D_id] += 1;

    // 获取观测图像、相机和二维特征点信息
    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // 对于不在主要优化列表中的图像，我们不希望优化其相机参数
    // 这些图像仅提供额外的观测约束，但其位姿和内参将保持固定
    if (camera_ids_.count(image.CameraId()) == 0) {
      // 记录相机ID并将其标记为固定相机
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamera(image.CameraId());
    }

    // 创建代价函数
    ceres::CostFunction* cost_function = nullptr;

    // 根据相机模型类型选择对应的代价函数
    // 注意：这里始终使用ConstantPoseCostFunction，因为这些额外观测的图像位姿被视为固定
    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                 \
    break;

      CAMERA_MODEL_SWITCH_CASES // 宏展开处理所有相机模型类型

#undef CAMERA_MODEL_CASE
    }

    // 添加残差块：只优化三维点坐标和相机内参（位姿固定）
    // 这里相机位姿被视为常量，只有三维点坐标和相机内参可能被优化
    problem_->AddResidualBlock(cost_function, loss_function,
                               point3D.XYZ().data(), camera.ParamsData());
  }
}

/**
 * [功能描述]：将激光雷达约束添加到三维点的优化问题中
 * @param [point3D_id]：要添加约束的三维点ID
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @param [loss_function]：Ceres优化器的损失函数，用于减轻异常值影响
 */
void BundleAdjuster::AddLidarToProblem(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  // 获取指定ID的三维点
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // 在激光雷达匹配映射表中查找该三维点的对应激光雷达点
  auto ptr = config_.lidar_maps_.find(point3D_id);
  if (ptr != config_.lidar_maps_.end()){
    // 获取对应激光雷达点的平面方程参数(ax+by+cz+d=0)
    Eigen::Matrix<double,4,1> abcd = ptr->second.LidarABCD();

    // 检查平面参数是否有效（不包含NaN值）
    for (int i = 0; i < 4; i++){
      if(std::isnan(abcd(i))){
        return; // 如果包含无效值，则放弃添加该约束
      } 
    }

    // 根据激光雷达点类型选择不同的约束权重
    double w;
    LidarPointType type = ptr->second.Type();
    if (type == LidarPointType::Proj){
        // 投影类型点约束权重
        w = options_.proj_lidar_constraint_weight;
    } else if (type == LidarPointType::Icp){
      // 普通ICP点约束权重
      w = options_.icp_lidar_constraint_weight;
    } else if (type == LidarPointType::IcpGround){
      // 地面ICP点约束权重（通常会给地面点更高权重）
      w = options_.icp_ground_lidar_constraint_weight;
    } else {
      // 未知类型，输出错误信息并放弃添加约束
      std::cout<<"This lidar point type is missing"<<std::endl;
      return;
    }

    // 创建激光雷达约束的代价函数
    ceres::CostFunction* cost_function = nullptr;
    cost_function =BundleAdjustmentLidarCostFunction::Create( 
            abcd,w);  // 使用平面参数和权重创建代价函数

    // 添加残差块：将激光雷达约束添加到三维点坐标
    // 注意：这里只优化三维点坐标，不涉及相机参数
    problem_->AddResidualBlock(cost_function, loss_function, point3D.XYZ().data());
  }
  
}

/**
 * [功能描述]：设置相机参数在优化过程中的参数化方式
 * 此函数决定哪些相机参数可以优化，哪些保持固定
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 */
void BundleAdjuster::ParameterizeCameras(Reconstruction* reconstruction) {
  // 根据优化选项判断是否所有相机内参都应该保持固定
  // 如果三个相机内参优化选项都关闭，则整个相机参数都将保持固定
  const bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;

  // 遍历所有在优化问题中涉及到的相机ID
  for (const camera_t camera_id : camera_ids_) {
    // 获取相机对象
    Camera& camera = reconstruction->Camera(camera_id);

    // 判断是否应将整个相机参数设为常量
    // 条件：全局设置所有相机固定 或 该特定相机被指定为固定
    if (constant_camera || config_.IsConstantCamera(camera_id)) {
      // 将相机参数块设为常量，不参与优化
      problem_->SetParameterBlockConstant(camera.ParamsData());
      continue; // 跳过后续处理
    } else {
      // 相机参数部分可变，需要确定哪些参数保持固定
      // 存储所有要保持固定的参数索引
      std::vector<int> const_camera_params;

      // 如果焦距不参与优化，将焦距相关参数索引添加到固定参数列表
      if (!options_.refine_focal_length) {
        const std::vector<size_t>& params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }

      // 如果主点不参与优化，将主点相关参数索引添加到固定参数列表
      if (!options_.refine_principal_point) {
        const std::vector<size_t>& params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }

      // 如果额外参数（如畸变系数）不参与优化，将其索引添加到固定参数列表
      if (!options_.refine_extra_params) {
        const std::vector<size_t>& params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }

      // 如果有需要固定的参数，设置参数子集流形
      // 子集流形允许参数块中的部分参数固定，部分参数可变
      if (const_camera_params.size() > 0) {
        SetSubsetManifold(static_cast<int>(camera.NumParams()),
                          const_camera_params, problem_.get(),
                          camera.ParamsData());
      }
    }
  }
}

/**
 * [功能描述]：设置三维点在优化问题中的参数化方式
 * 此函数决定哪些三维点坐标可以优化，哪些需要保持固定
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 */
void BundleAdjuster::ParameterizePoints(Reconstruction* reconstruction) {
  // 第一部分：处理部分观测被添加到问题中的三维点
  // 遍历所有已添加到优化问题中的三维点观测记录
  for (const auto elem : point3D_num_observations_) {
    // 获取三维点ID和对象
    Point3D& point3D = reconstruction->Point3D(elem.first);

    // 判断条件：如果三维点的实际观测轨迹长度大于在当前优化问题中添加的观测数
    // 这意味着该点有部分观测没有添加到优化问题中，通常是因为观测它的某些图像不在优化范围内
    if (point3D.Track().Length() > elem.second) {
      // 将这类三维点设为常量，不参与优化
      // 这是为了避免优化时只考虑部分观测导致的结果偏差
      problem_->SetParameterBlockConstant(point3D.XYZ().data());
    }
  }

  // 第二部分：处理用户明确指定为常量的三维点
  // 遍历配置中被标记为常量的所有三维点ID
  for (const point3D_t point3D_id : config_.ConstantPoints()) {
    // 获取三维点对象
    Point3D& point3D = reconstruction->Point3D(point3D_id);
    // 将该三维点设为常量，不参与优化
    problem_->SetParameterBlockConstant(point3D.XYZ().data());
  }
}

////////////////////////////////////////////////////////////////////////////////
// ParallelBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

bool ParallelBundleAdjuster::Options::Check() const {
  CHECK_OPTION_GE(max_num_iterations, 0);
  return true;
}

ParallelBundleAdjuster::ParallelBundleAdjuster(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const BundleAdjustmentConfig& config)
    : options_(options),
      ba_options_(ba_options),
      config_(config),
      num_measurements_(0) {
  CHECK(options_.Check());
  CHECK(ba_options_.Check());
  CHECK_EQ(config_.NumConstantCameras(), 0)
      << "PBA does not allow to set individual cameras constant";
  CHECK_EQ(config_.NumConstantPoses(), 0)
      << "PBA does not allow to set individual translational elements constant";
  CHECK_EQ(config_.NumConstantTvecs(), 0)
      << "PBA does not allow to set individual translational elements constant";
  CHECK(config_.NumVariablePoints() == 0 && config_.NumConstantPoints() == 0)
      << "PBA does not allow to parameterize individual 3D points";
}

/**
 * [功能描述]：并行光束法平差(PBA)的主求解函数
 * 使用GPU或多线程CPU加速大规模光束法平差计算
 * @param [reconstruction]：重建对象指针，包含整个三维重建场景的数据
 * @return [bool]：优化是否成功执行
 */
bool ParallelBundleAdjuster::Solve(Reconstruction* reconstruction) {
  // 检查reconstruction指针非空
  CHECK_NOTNULL(reconstruction);

  // 确保PBA没有被重复使用（num_measurements_初始为0）
  CHECK_EQ(num_measurements_, 0)
      << "Cannot use the same ParallelBundleAdjuster multiple times";

  // PBA不支持优化主点参数
  CHECK(!ba_options_.refine_principal_point);

  // PBA要求焦距和额外参数必须同时优化或同时不优化
  CHECK_EQ(ba_options_.refine_focal_length, ba_options_.refine_extra_params);

  // 准备优化数据，收集相机、三维点和观测数据
  SetUp(reconstruction);

  // 计算总残差数量（每个观测产生2个残差：x和y方向）
  const int num_residuals = static_cast<int>(2 * measurements_.size());

  // 线程数设置：如果残差数少于阈值，使用单线程
  size_t num_threads = options_.num_threads;
  if (num_residuals < options_.min_num_residuals_for_multi_threading) {
    num_threads = 1;
  }

  // 设备选择：根据残差数量和用户设置选择计算设备
  pba::ParallelBA::DeviceT device;
  const int kMaxNumResidualsFloat = 100 * 1000;
  if (num_residuals > kMaxNumResidualsFloat) {
    // 残差数量大时使用CPU双精度（更稳定但较慢）
    // 这个阈值是经验选择的，确保系统能可靠求解
    device = pba::ParallelBA::PBA_CPU_DOUBLE;
  } else {
    // 残差数量适中时可以使用GPU加速
    if (options_.gpu_index < 0) {
      // 使用默认GPU设备
      device = pba::ParallelBA::PBA_CUDA_DEVICE_DEFAULT;
    } else {
      // 使用指定的GPU设备
      device = static_cast<pba::ParallelBA::DeviceT>(
          pba::ParallelBA::PBA_CUDA_DEVICE0 + options_.gpu_index);
    }
  }

  // 创建并配置ParallelBA对象
  pba::ParallelBA pba(device, num_threads);

  // 设置为完整的光束法调整模式（同时优化相机参数和三维点）
  pba.SetNextBundleMode(pba::ParallelBA::BUNDLE_FULL);
  // 启用径向畸变
  pba.EnableRadialDistortion(pba::ParallelBA::PBA_PROJECTION_DISTORTION);
  // 设置相机内参是否固定（基于BA选项）
  pba.SetFixedIntrinsics(!ba_options_.refine_focal_length &&
                         !ba_options_.refine_extra_params);

  // 配置PBA的优化参数
  pba::ConfigBA* pba_config = pba.GetInternalConfig();
  // 降低LM算法的阈值，使收敛更严格
  pba_config->__lm_delta_threshold /= 100.0f;
  pba_config->__lm_gradient_threshold /= 100.0f;
  // 设置MSE阈值为0，使优化一直运行到最大迭代次数
  pba_config->__lm_mse_threshold = 0.0f;
  // 设置共轭梯度法最小迭代次数
  pba_config->__cg_min_iteration = 10;
  // 设置详细级别为中等
  pba_config->__verbose_level = 2;
  // 设置最大迭代次数
  pba_config->__lm_max_iteration = options_.max_num_iterations;

  // 传入优化数据
  // 相机数据
  pba.SetCameraData(cameras_.size(), cameras_.data());
  // 三维点数据
  pba.SetPointData(points3D_.size(), points3D_.data());
  // 观测（投影）数据
  pba.SetProjection(measurements_.size(), measurements_.data(),
                    point3D_idxs_.data(), camera_idxs_.data());

  // 计时并运行光束法平差
  Timer timer;
  timer.Start();
  pba.RunBundleAdjustment();
  timer.Pause();

  // 从PBA结果构造Ceres求解器摘要信息
  summary_.num_residuals_reduced = num_residuals;
  // 计算有效参数数量：8个相机参数 × 图像数 - 2 × 固定相机数 + 3 × 三维点数
  summary_.num_effective_parameters_reduced =
      static_cast<int>(8 * config_.NumImages() -
                       2 * config_.NumConstantCameras() + 3 * points3D_.size());
  // 记录成功步数为LM迭代次数+1
  summary_.num_successful_steps = pba_config->GetIterationsLM() + 1;
  // 设置终止类型为用户成功（自定义终止）
  summary_.termination_type = ceres::TerminationType::USER_SUCCESS;
  // 计算初始代价和最终代价（从MSE换算）
  summary_.initial_cost =
      pba_config->GetInitialMSE() * summary_.num_residuals_reduced / 4;
  summary_.final_cost =
      pba_config->GetFinalMSE() * summary_.num_residuals_reduced / 4;
  // 记录总运行时间
  summary_.total_time_in_seconds = timer.ElapsedSeconds();

  // 将优化后的相机和三维点数据从PBA格式转回COLMAP格式
  TearDown(reconstruction);

  // 如果设置了打印摘要，则输出优化报告
  if (options_.print_summary) {
    PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  return true;
}

const ceres::Solver::Summary& ParallelBundleAdjuster::Summary() const {
  return summary_;
}

/**
 * 检查并行束调整(PBA)是否支持当前的优化选项
 * 并行束调整有一些特定限制，不是所有场景配置都支持
 * @param options: 束调整的优化配置选项
 * @param reconstruction: 三维重建场景的数据结构
 * @return 如果PBA支持当前选项，返回true；否则返回false
 */
bool ParallelBundleAdjuster::IsSupported(const BundleAdjustmentOptions& options,
                                         const Reconstruction& reconstruction) {
  // 检查条件1：不支持优化主点参数(principal point)
  // 检查条件2：焦距和额外参数(如畸变系数)必须同时优化或同时不优化
  if (options.refine_principal_point ||
      options.refine_focal_length != options.refine_extra_params) {
    return false; // 不满足条件，返回不支持
  }

  // 检查条件3：所有相机必须使用SIMPLE_RADIAL模型，且不能共享内参
  // SIMPLE_RADIAL是一种简单的相机模型，只包含焦距、主点和一个径向畸变参数
  std::set<camera_t> camera_ids; // 用于跟踪已处理的相机ID，确保不重复
  for (const auto& image : reconstruction.Images()) { // 遍历所有图像
    if (image.second.IsRegistered()) { // 只检查已注册的图像
      // 检查该相机ID是否已经出现过(共享内参检查)
      // 同时检查相机模型是否为SIMPLE_RADIAL
      if (camera_ids.count(image.second.CameraId()) != 0 ||
          reconstruction.Camera(image.second.CameraId()).ModelId() !=
              SimpleRadialCameraModel::model_id) {
        return false; // 不满足条件，返回不支持
      }
      // 将当前相机ID添加到已处理集合中
      camera_ids.insert(image.second.CameraId());
    }
  }
  return true;
}

void ParallelBundleAdjuster::SetUp(Reconstruction* reconstruction) {
  // Important: PBA requires the track of 3D points to be stored
  // contiguously, i.e. the point3D_idxs_ vector contains consecutive indices.
  cameras_.reserve(config_.NumImages());
  camera_ids_.reserve(config_.NumImages());
  ordered_image_ids_.reserve(config_.NumImages());
  image_id_to_camera_idx_.reserve(config_.NumImages());
  AddImagesToProblem(reconstruction);
  AddPointsToProblem(reconstruction);
}

void ParallelBundleAdjuster::TearDown(Reconstruction* reconstruction) {
  for (size_t i = 0; i < cameras_.size(); ++i) {
    const image_t image_id = ordered_image_ids_[i];
    const pba::CameraT& pba_camera = cameras_[i];

    // Note: Do not use PBA's quaternion methods as they seem to lead to
    // numerical instability or other issues.
    Image& image = reconstruction->Image(image_id);
    Eigen::Matrix3d rotation_matrix;
    pba_camera.GetMatrixRotation(rotation_matrix.data());
    pba_camera.GetTranslation(image.Tvec().data());
    image.Qvec() = RotationMatrixToQuaternion(rotation_matrix.transpose());

    Camera& camera = reconstruction->Camera(image.CameraId());
    camera.Params(0) = pba_camera.GetFocalLength();
    camera.Params(3) = pba_camera.GetProjectionDistortion();
  }

  for (size_t i = 0; i < points3D_.size(); ++i) {
    Point3D& point3D = reconstruction->Point3D(ordered_point3D_ids_[i]);
    points3D_[i].GetPoint(point3D.XYZ().data());
  }
}

void ParallelBundleAdjuster::AddImagesToProblem(
    Reconstruction* reconstruction) {
  for (const image_t image_id : config_.Images()) {
    const Image& image = reconstruction->Image(image_id);
    CHECK_EQ(camera_ids_.count(image.CameraId()), 0)
        << "PBA does not support shared intrinsics";

    const Camera& camera = reconstruction->Camera(image.CameraId());
    CHECK_EQ(camera.ModelId(), SimpleRadialCameraModel::model_id)
        << "PBA only supports the SIMPLE_RADIAL camera model";

    // Note: Do not use PBA's quaternion methods as they seem to lead to
    // numerical instability or other issues.
    const Eigen::Matrix3d rotation_matrix =
        QuaternionToRotationMatrix(image.Qvec()).transpose();

    pba::CameraT pba_camera;
    pba_camera.SetFocalLength(camera.Params(0));
    pba_camera.SetProjectionDistortion(camera.Params(3));
    pba_camera.SetMatrixRotation(rotation_matrix.data());
    pba_camera.SetTranslation(image.Tvec().data());

    CHECK(!config_.HasConstantTvec(image_id))
        << "PBA cannot fix partial extrinsics";
    if (!ba_options_.refine_extrinsics || config_.HasConstantPose(image_id)) {
      CHECK(config_.IsConstantCamera(image.CameraId()))
          << "PBA cannot fix extrinsics only";
      pba_camera.SetConstantCamera();
    } else if (config_.IsConstantCamera(image.CameraId())) {
      pba_camera.SetFixedIntrinsic();
    } else {
      pba_camera.SetVariableCamera();
    }

    num_measurements_ += image.NumPoints3D();
    cameras_.push_back(pba_camera);
    camera_ids_.insert(image.CameraId());
    ordered_image_ids_.push_back(image_id);
    image_id_to_camera_idx_.emplace(image_id,
                                    static_cast<int>(cameras_.size()) - 1);

    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point3D_ids_.insert(point2D.Point3DId());
      }
    }
  }
}

void ParallelBundleAdjuster::AddPointsToProblem(
    Reconstruction* reconstruction) {
  points3D_.resize(point3D_ids_.size());
  ordered_point3D_ids_.resize(point3D_ids_.size());
  measurements_.resize(num_measurements_);
  camera_idxs_.resize(num_measurements_);
  point3D_idxs_.resize(num_measurements_);

  int point3D_idx = 0;
  size_t measurement_idx = 0;

  for (const auto point3D_id : point3D_ids_) {
    const Point3D& point3D = reconstruction->Point3D(point3D_id);
    points3D_[point3D_idx].SetPoint(point3D.XYZ().data());
    ordered_point3D_ids_[point3D_idx] = point3D_id;

    for (const auto track_el : point3D.Track().Elements()) {
      if (image_id_to_camera_idx_.count(track_el.image_id) > 0) {
        const Image& image = reconstruction->Image(track_el.image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        measurements_[measurement_idx].SetPoint2D(
            point2D.X() - camera.Params(1), point2D.Y() - camera.Params(2));
        camera_idxs_[measurement_idx] =
            image_id_to_camera_idx_.at(track_el.image_id);
        point3D_idxs_[measurement_idx] = point3D_idx;
        measurement_idx += 1;
      }
    }
    point3D_idx += 1;
  }

  CHECK_EQ(point3D_idx, points3D_.size());
  CHECK_EQ(measurement_idx, measurements_.size());
}

////////////////////////////////////////////////////////////////////////////////
// RigBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

RigBundleAdjuster::RigBundleAdjuster(const BundleAdjustmentOptions& options,
                                     const Options& rig_options,
                                     const BundleAdjustmentConfig& config)
    : BundleAdjuster(options, config), rig_options_(rig_options) {}

bool RigBundleAdjuster::Solve(Reconstruction* reconstruction,
                              std::vector<CameraRig>* camera_rigs) {
  CHECK_NOTNULL(reconstruction);
  CHECK_NOTNULL(camera_rigs);
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  // Check the validity of the provided camera rigs.
  std::unordered_set<camera_t> rig_camera_ids;
  for (auto& camera_rig : *camera_rigs) {
    camera_rig.Check(*reconstruction);
    for (const auto& camera_id : camera_rig.GetCameraIds()) {
      CHECK_EQ(rig_camera_ids.count(camera_id), 0)
          << "Camera must not be part of multiple camera rigs";
      rig_camera_ids.insert(camera_id);
    }

    for (const auto& snapshot : camera_rig.Snapshots()) {
      for (const auto& image_id : snapshot) {
        CHECK_EQ(image_id_to_camera_rig_.count(image_id), 0)
            << "Image must not be part of multiple camera rigs";
        image_id_to_camera_rig_.emplace(image_id, &camera_rig);
      }
    }
  }

  problem_ = std::make_unique<ceres::Problem>();

  ceres::LossFunction* loss_function = options_.CreateLossFunction();
  SetUp(reconstruction, camera_rigs, loss_function);

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  solver_options.num_threads =
      GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Rig Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction, *camera_rigs);

  return true;
}

void RigBundleAdjuster::SetUp(Reconstruction* reconstruction,
                              std::vector<CameraRig>* camera_rigs,
                              ceres::LossFunction* loss_function) {
  ComputeCameraRigPoses(*reconstruction, *camera_rigs);

  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, camera_rigs, loss_function);
  }

  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
  ParameterizeCameraRigs(reconstruction);
}

void RigBundleAdjuster::TearDown(Reconstruction* reconstruction,
                                 const std::vector<CameraRig>& camera_rigs) {
  for (const auto& elem : image_id_to_camera_rig_) {
    const auto image_id = elem.first;
    const auto& camera_rig = *elem.second;
    auto& image = reconstruction->Image(image_id);
    ConcatenatePoses(*image_id_to_rig_qvec_.at(image_id),
                     *image_id_to_rig_tvec_.at(image_id),
                     camera_rig.RelativeQvec(image.CameraId()),
                     camera_rig.RelativeTvec(image.CameraId()), &image.Qvec(),
                     &image.Tvec());
  }
}

void RigBundleAdjuster::AddImageToProblem(const image_t image_id,
                                          Reconstruction* reconstruction,
                                          std::vector<CameraRig>* camera_rigs,
                                          ceres::LossFunction* loss_function) {
  const double max_squared_reproj_error =
      rig_options_.max_reproj_error * rig_options_.max_reproj_error;

  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  const bool constant_pose = config_.HasConstantPose(image_id);
  const bool constant_tvec = config_.HasConstantTvec(image_id);

  double* qvec_data = nullptr;
  double* tvec_data = nullptr;
  double* rig_qvec_data = nullptr;
  double* rig_tvec_data = nullptr;
  double* camera_params_data = camera.ParamsData();
  CameraRig* camera_rig = nullptr;
  Eigen::Matrix3x4d rig_proj_matrix = Eigen::Matrix3x4d::Zero();

  if (image_id_to_camera_rig_.count(image_id) > 0) {
    CHECK(!constant_pose)
        << "Images contained in a camera rig must not have constant pose";
    CHECK(!constant_tvec)
        << "Images contained in a camera rig must not have constant tvec";
    camera_rig = image_id_to_camera_rig_.at(image_id);
    rig_qvec_data = image_id_to_rig_qvec_.at(image_id)->data();
    rig_tvec_data = image_id_to_rig_tvec_.at(image_id)->data();
    qvec_data = camera_rig->RelativeQvec(image.CameraId()).data();
    tvec_data = camera_rig->RelativeTvec(image.CameraId()).data();

    // Concatenate the absolute pose of the rig and the relative pose the camera
    // within the rig to detect outlier observations.
    Eigen::Vector4d rig_concat_qvec;
    Eigen::Vector3d rig_concat_tvec;
    ConcatenatePoses(*image_id_to_rig_qvec_.at(image_id),
                     *image_id_to_rig_tvec_.at(image_id),
                     camera_rig->RelativeQvec(image.CameraId()),
                     camera_rig->RelativeTvec(image.CameraId()),
                     &rig_concat_qvec, &rig_concat_tvec);
    rig_proj_matrix = ComposeProjectionMatrix(rig_concat_qvec, rig_concat_tvec);
  } else {
    // CostFunction assumes unit quaternions.
    image.NormalizeQvec();
    qvec_data = image.Qvec().data();
    tvec_data = image.Tvec().data();
  }

  // Collect cameras for final parameterization.
  CHECK(image.HasCamera());
  camera_ids_.insert(image.CameraId());

  // The number of added observations for the current image.
  size_t num_observations = 0;

  // Add residuals to bundle adjustment problem.
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }

    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    assert(point3D.Track().Length() > 1);

    if (camera_rig != nullptr &&
        CalculateSquaredReprojectionError(point2D.XY(), point3D.XYZ(),
                                          rig_proj_matrix,
                                          camera) > max_squared_reproj_error) {
      continue;
    }

    num_observations += 1;
    point3D_num_observations_[point2D.Point3DId()] += 1;

    ceres::CostFunction* cost_function = nullptr;

    if (camera_rig == nullptr) {
      if (constant_pose) {
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                 \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }

        problem_->AddResidualBlock(cost_function, loss_function,
                                   point3D.XYZ().data(), camera_params_data);
      } else {
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::kModelId:                                            \
    cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }

        problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                   tvec_data, point3D.XYZ().data(),
                                   camera_params_data);
      }
    } else {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case CameraModel::kModelId:                                               \
    cost_function =                                                         \
        RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
                                                                            \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
      problem_->AddResidualBlock(cost_function, loss_function, rig_qvec_data,
                                 rig_tvec_data, qvec_data, tvec_data,
                                 point3D.XYZ().data(), camera_params_data);
    }
  }

  if (num_observations > 0) {
    parameterized_qvec_data_.insert(qvec_data);

    if (camera_rig != nullptr) {
      parameterized_qvec_data_.insert(rig_qvec_data);

      // Set the relative pose of the camera constant if relative pose
      // refinement is disabled or if it is the reference camera to avoid over-
      // parameterization of the camera pose.
      if (!rig_options_.refine_relative_poses ||
          image.CameraId() == camera_rig->RefCameraId()) {
        problem_->SetParameterBlockConstant(qvec_data);
        problem_->SetParameterBlockConstant(tvec_data);
      }
    }

    // Set pose parameterization.
    if (!constant_pose && constant_tvec) {
      const std::vector<int>& constant_tvec_idxs =
          config_.ConstantTvec(image_id);
      SetSubsetManifold(3, constant_tvec_idxs, problem_.get(), tvec_data);
    }
  }
}

void RigBundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                          Reconstruction* reconstruction,
                                          ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
    return;
  }

  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `AddImageToProblem`.
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    point3D_num_observations_[point3D_id] += 1;

    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_ids_.count(image.CameraId()) == 0) {
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamera(image.CameraId());
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::kModelId:                                              \
    cost_function =                                                        \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(     \
            image.Qvec(), image.Tvec(), point2D.XY());                     \
    problem_->AddResidualBlock(cost_function, loss_function,               \
                               point3D.XYZ().data(), camera.ParamsData()); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
  }
}

void RigBundleAdjuster::ComputeCameraRigPoses(
    const Reconstruction& reconstruction,
    const std::vector<CameraRig>& camera_rigs) {
  camera_rig_qvecs_.reserve(camera_rigs.size());
  camera_rig_tvecs_.reserve(camera_rigs.size());
  for (const auto& camera_rig : camera_rigs) {
    camera_rig_qvecs_.emplace_back();
    camera_rig_tvecs_.emplace_back();
    auto& rig_qvecs = camera_rig_qvecs_.back();
    auto& rig_tvecs = camera_rig_tvecs_.back();
    rig_qvecs.resize(camera_rig.NumSnapshots());
    rig_tvecs.resize(camera_rig.NumSnapshots());
    for (size_t snapshot_idx = 0; snapshot_idx < camera_rig.NumSnapshots();
         ++snapshot_idx) {
      camera_rig.ComputeAbsolutePose(snapshot_idx, reconstruction,
                                     &rig_qvecs[snapshot_idx],
                                     &rig_tvecs[snapshot_idx]);
      for (const auto image_id : camera_rig.Snapshots()[snapshot_idx]) {
        image_id_to_rig_qvec_.emplace(image_id, &rig_qvecs[snapshot_idx]);
        image_id_to_rig_tvec_.emplace(image_id, &rig_tvecs[snapshot_idx]);
      }
    }
  }
}

void RigBundleAdjuster::ParameterizeCameraRigs(Reconstruction* reconstruction) {
  for (double* qvec_data : parameterized_qvec_data_) {
    SetQuaternionManifold(problem_.get(), qvec_data);
  }
}

void PrintSolverSummary(const ceres::Solver::Summary& summary) {
  std::cout << std::right << std::setw(16) << "Residuals : ";
  std::cout << std::left << summary.num_residuals_reduced << std::endl;

  std::cout << std::right << std::setw(16) << "Parameters : ";
  std::cout << std::left << summary.num_effective_parameters_reduced
            << std::endl;

  std::cout << std::right << std::setw(16) << "Iterations : ";
  std::cout << std::left
            << summary.num_successful_steps + summary.num_unsuccessful_steps
            << std::endl;

  std::cout << std::right << std::setw(16) << "Time : ";
  std::cout << std::left << summary.total_time_in_seconds << " [s]"
            << std::endl;

  std::cout << std::right << std::setw(16) << "Initial cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Final cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Termination : ";

  std::string termination = "";

  switch (summary.termination_type) {
    case ceres::CONVERGENCE:
      termination = "Convergence";
      break;
    case ceres::NO_CONVERGENCE:
      termination = "No convergence";
      break;
    case ceres::FAILURE:
      termination = "Failure";
      break;
    case ceres::USER_SUCCESS:
      termination = "User success";
      break;
    case ceres::USER_FAILURE:
      termination = "User failure";
      break;
    default:
      termination = "Unknown";
      break;
  }

  std::cout << std::right << termination << std::endl;
  std::cout << std::endl;
}

}  // namespace colmap
