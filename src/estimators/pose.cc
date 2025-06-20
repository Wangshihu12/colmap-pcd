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

#include "estimators/pose.h"

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/essential_matrix.h"
#include "base/pose.h"
#include "estimators/absolute_pose.h"
#include "estimators/essential_matrix.h"
#include "optim/bundle_adjustment.h"
#include "util/matrix.h"
#include "util/misc.h"
#include "util/threading.h"

namespace colmap {
namespace {

typedef LORANSAC<P3PEstimator, EPNPEstimator> AbsolutePoseRANSAC;

void EstimateAbsolutePoseKernel(const Camera& camera,
                                const double focal_length_factor,
                                const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D,
                                const RANSACOptions& options,
                                AbsolutePoseRANSAC::Report* report) {
  // Scale the focal length by the given factor.
  Camera scaled_camera = camera;
  const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
  for (const size_t idx : focal_length_idxs) {
    scaled_camera.Params(idx) *= focal_length_factor;
  }

  // Normalize image coordinates with current camera hypothesis.
  std::vector<Eigen::Vector2d> points2D_N(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
  }

  // Estimate pose for given focal length.
  auto custom_options = options;
  custom_options.max_error =
      scaled_camera.ImageToWorldThreshold(options.max_error);
  AbsolutePoseRANSAC ransac(custom_options);
  *report = ransac.Estimate(points2D_N, points3D);
}

}  // namespace

/**
 * 绝对位姿估计函数（PnP问题求解）
 * 
 * 该函数解决Perspective-n-Point (PnP)问题，即通过已知的2D-3D点对应关系
 * 估计相机的绝对位姿（位置和姿态）。支持同时估计焦距参数和使用RANSAC
 * 进行鲁棒估计以处理外点。
 * 
 * @param options 绝对位姿估计的配置选项
 * @param points2D 图像中的2D观测点
 * @param points3D 对应的3D世界坐标点
 * @param qvec 输出相机姿态四元数 [w, x, y, z]
 * @param tvec 输出相机位置向量 [x, y, z]
 * @param camera 相机参数对象，可能被修改(如果估计焦距)
 * @param num_inliers 输出内点数量
 * @param inlier_mask 输出内点掩码，标记哪些点对是内点
 * @return 估计成功返回true，失败返回false
 */
bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                          Camera* camera, size_t* num_inliers,
                          std::vector<char>* inlier_mask) {
  // 检查输入参数的有效性
  options.Check();

  // 生成焦距因子列表
  std::vector<double> focal_length_factors;
  
  if (options.estimate_focal_length) {
    // 如果需要估计焦距，生成一系列焦距因子进行测试
    // 使用二次函数生成更多的小焦距样本，因为小焦距通常更敏感
    focal_length_factors.reserve(options.num_focal_length_samples + 1);
    const double fstep = 1.0 / options.num_focal_length_samples;  // 步长
    const double fscale =
        options.max_focal_length_ratio - options.min_focal_length_ratio;  // 焦距范围
    
    // 使用二次函数 f*f 生成焦距因子，使得更多样本集中在小焦距区域
    for (double f = 0; f <= 1.0; f += fstep) {
      focal_length_factors.push_back(options.min_focal_length_ratio +
                                     fscale * f * f);
    }
  } else {
    // 如果不估计焦距，只使用当前焦距(因子为1)
    focal_length_factors.reserve(1);
    focal_length_factors.push_back(1);
  }

  // 准备多线程处理
  // 为每个焦距因子创建一个future对象，用于异步处理
  std::vector<std::future<void>> futures;
  futures.resize(focal_length_factors.size());
  
  // 为每个焦距因子创建一个报告对象，存储RANSAC结果
  std::vector<typename AbsolutePoseRANSAC::Report,
              Eigen::aligned_allocator<typename AbsolutePoseRANSAC::Report>>
      reports;
  reports.resize(focal_length_factors.size());

  // 创建线程池，线程数为用户指定数量和焦距因子数量的最小值
  // 这避免了创建过多线程导致的资源浪费
  ThreadPool thread_pool(std::min(
      options.num_threads, static_cast<int>(focal_length_factors.size())));

  // 为每个焦距因子启动一个异步任务
  for (size_t i = 0; i < focal_length_factors.size(); ++i) {
    futures[i] = thread_pool.AddTask(
        EstimateAbsolutePoseKernel,    // 核心位姿估计函数
        *camera,                       // 相机参数
        focal_length_factors[i],       // 当前焦距因子
        points2D,                      // 2D观测点
        points3D,                      // 3D世界点
        options.ransac_options,        // RANSAC参数
        &reports[i]);                  // 输出报告
  }

  // 初始化最佳结果变量
  double focal_length_factor = 0;    // 最佳焦距因子
  Eigen::Matrix3x4d proj_matrix;     // 最佳投影矩阵 [R|t]
  *num_inliers = 0;                  // 最佳内点数量
  inlier_mask->clear();              // 清空内点掩码

  // 等待所有线程完成，并找到最佳模型
  for (size_t i = 0; i < focal_length_factors.size(); ++i) {
    futures[i].get();  // 等待第i个任务完成
    const auto report = reports[i];  // 获取第i个任务的结果
    
    // 如果当前任务成功且内点数量更多，更新最佳结果
    if (report.success && report.support.num_inliers > *num_inliers) {
      *num_inliers = report.support.num_inliers;  // 更新最佳内点数量
      proj_matrix = report.model;                 // 更新最佳投影矩阵
      *inlier_mask = report.inlier_mask;          // 更新内点掩码
      focal_length_factor = focal_length_factors[i];  // 记录最佳焦距因子
    }
  }

  // 如果没有找到任何有效的内点，估计失败
  if (*num_inliers == 0) {
    return false;
  }

  // 如果估计了焦距且找到了有效解，更新相机参数
  if (options.estimate_focal_length && *num_inliers > 0) {
    // 获取相机参数中焦距参数的索引
    const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
    
    // 将最佳焦距因子应用到相机的焦距参数上
    for (const size_t idx : focal_length_idxs) {
      camera->Params(idx) *= focal_length_factor;
    }
  }

  // 从投影矩阵中提取位姿参数
  // 投影矩阵的格式为 [R|t]，其中R是3x3旋转矩阵，t是3x1平移向量
  *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());  // 提取旋转部分并转换为四元数
  *tvec = proj_matrix.rightCols<1>();                             // 提取平移部分

  // 检查结果是否包含NaN值
  if (IsNaN(*qvec) || IsNaN(*tvec)) {
    return false;
  }

  // 位姿估计成功
  return true;
}

size_t EstimateRelativePose(const RANSACOptions& ransac_options,
                            const std::vector<Eigen::Vector2d>& points1,
                            const std::vector<Eigen::Vector2d>& points2,
                            Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) {
  RANSAC<EssentialMatrixFivePointEstimator> ransac(ransac_options);
  const auto report = ransac.Estimate(points1, points2);

  if (!report.success) {
    return 0;
  }

  std::vector<Eigen::Vector2d> inliers1(report.support.num_inliers);
  std::vector<Eigen::Vector2d> inliers2(report.support.num_inliers);

  size_t j = 0;
  for (size_t i = 0; i < points1.size(); ++i) {
    if (report.inlier_mask[i]) {
      inliers1[j] = points1[i];
      inliers2[j] = points2[i];
      j += 1;
    }
  }

  Eigen::Matrix3d R;

  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(report.model, inliers1, inliers2, &R, tvec,
                          &points3D);

  *qvec = RotationMatrixToQuaternion(R);

  if (IsNaN(*qvec) || IsNaN(*tvec)) {
    return 0;
  }

  return points3D.size();
}

bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        Camera* camera, Eigen::Matrix6d* rot_tvec_covariance) {
  CHECK_EQ(inlier_mask.size(), points2D.size());
  CHECK_EQ(points2D.size(), points3D.size());
  options.Check();

  ceres::LossFunction* loss_function =
      new ceres::CauchyLoss(options.loss_function_scale);

  double* qvec_data = qvec->data();
  double* tvec_data = tvec->data();
  ceres::Problem problem;

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera->ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case CameraModel::kModelId:                                             \
    cost_function =                                                       \
        BundleAdjustmentConstantPoint3DCostFunction<CameraModel>::Create( \
            points2D[i], points3D[i]);                                    \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    problem.AddResidualBlock(cost_function, loss_function, qvec->data(),
                             tvec->data(), camera->ParamsData());
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, qvec->data());

    // Camera parameterization.
    if (!options.refine_focal_length && !options.refine_extra_params) {
      problem.SetParameterBlockConstant(camera->ParamsData());
    } else {
      // Always set the principal point as fixed.
      std::vector<int> camera_params_const;
      const std::vector<size_t>& principal_point_idxs =
          camera->PrincipalPointIdxs();
      camera_params_const.insert(camera_params_const.end(),
                                 principal_point_idxs.begin(),
                                 principal_point_idxs.end());

      if (!options.refine_focal_length) {
        const std::vector<size_t>& focal_length_idxs =
            camera->FocalLengthIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   focal_length_idxs.begin(),
                                   focal_length_idxs.end());
      }

      if (!options.refine_extra_params) {
        const std::vector<size_t>& extra_params_idxs =
            camera->ExtraParamsIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   extra_params_idxs.begin(),
                                   extra_params_idxs.end());
      }

      if (camera_params_const.size() == camera->NumParams()) {
        problem.SetParameterBlockConstant(camera->ParamsData());
      } else {
        SetSubsetManifold(static_cast<int>(camera->NumParams()),
                          camera_params_const, &problem, camera->ParamsData());
      }
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;

  // The overhead of creating threads is too large.
  solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options.print_summary) {
    PrintHeading2("Pose refinement report");
    PrintSolverSummary(summary);
  }

  if (problem.NumResiduals() > 0 && rot_tvec_covariance != nullptr) {
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);
    std::vector<const double*> parameter_blocks = {qvec_data, tvec_data};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      return false;
    }
    // The rotation covariance is estimated in the tangent space of the
    // quaternion, which corresponds to the 3-DoF axis-angle local
    // parameterization.
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 rot_tvec_covariance->data());
  }

  return summary.IsSolutionUsable();
}

bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) {
  CHECK_EQ(points1.size(), points2.size());

  // CostFunction assumes unit quaternions.
  *qvec = NormalizeQuaternion(*qvec);

  const double kMaxL2Error = 1.0;
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(kMaxL2Error);

  ceres::Problem problem;

  for (size_t i = 0; i < points1.size(); ++i) {
    ceres::CostFunction* cost_function =
        RelativePoseCostFunction::Create(points1[i], points2[i]);
    problem.AddResidualBlock(cost_function, loss_function, qvec->data(),
                             tvec->data());
  }

  SetQuaternionManifold(&problem, qvec->data());
  SetSphereManifold<3>(&problem, tvec->data());

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return summary.IsSolutionUsable();
}

bool RefineGeneralizedAbsolutePose(
    const AbsolutePoseRefinementOptions& options,
    const std::vector<char>& inlier_mask,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Eigen::Vector4d>& rig_qvecs,
    const std::vector<Eigen::Vector3d>& rig_tvecs, Eigen::Vector4d* qvec,
    Eigen::Vector3d* tvec, std::vector<Camera>* cameras,
    Eigen::Matrix6d* rot_tvec_covariance) {
  CHECK_EQ(points2D.size(), inlier_mask.size());
  CHECK_EQ(points2D.size(), points3D.size());
  CHECK_EQ(points2D.size(), camera_idxs.size());
  CHECK_EQ(rig_qvecs.size(), rig_tvecs.size());
  CHECK_EQ(rig_qvecs.size(), cameras->size());
  CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);
  CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()),
           cameras->size());
  options.Check();

  ceres::LossFunction* loss_function =
      new ceres::CauchyLoss(options.loss_function_scale);

  std::vector<double*> cameras_params_data;
  for (size_t i = 0; i < cameras->size(); i++) {
    cameras_params_data.push_back(cameras->at(i).ParamsData());
  }
  std::vector<size_t> camera_counts(cameras->size(), 0);
  double* qvec_data = qvec->data();
  double* tvec_data = tvec->data();

  std::vector<Eigen::Vector3d> points3D_copy = points3D;
  std::vector<Eigen::Vector4d> rig_qvecs_copy = rig_qvecs;
  std::vector<Eigen::Vector3d> rig_tvecs_copy = rig_tvecs;

  ceres::Problem problem;

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }
    const size_t camera_idx = camera_idxs[i];
    camera_counts[camera_idx] += 1;

    ceres::CostFunction* cost_function = nullptr;
    switch (cameras->at(camera_idx).ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::kModelId:                                              \
    cost_function =                                                        \
        RigBundleAdjustmentCostFunction<CameraModel>::Create(points2D[i]); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    problem.AddResidualBlock(
        cost_function, loss_function, qvec_data, tvec_data,
        rig_qvecs_copy[camera_idx].data(), rig_tvecs_copy[camera_idx].data(),
        points3D_copy[i].data(), cameras_params_data[camera_idx]);
    problem.SetParameterBlockConstant(points3D_copy[i].data());
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, qvec_data);

    // Camera parameterization.
    for (size_t i = 0; i < cameras->size(); i++) {
      if (camera_counts[i] == 0) continue;
      Camera& camera = cameras->at(i);

      // We don't optimize the rig parameters (it's likely under-constrained)
      problem.SetParameterBlockConstant(rig_qvecs_copy[i].data());
      problem.SetParameterBlockConstant(rig_tvecs_copy[i].data());

      if (!options.refine_focal_length && !options.refine_extra_params) {
        problem.SetParameterBlockConstant(camera.ParamsData());
      } else {
        // Always set the principal point as fixed.
        std::vector<int> camera_params_const;
        const std::vector<size_t>& principal_point_idxs =
            camera.PrincipalPointIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   principal_point_idxs.begin(),
                                   principal_point_idxs.end());

        if (!options.refine_focal_length) {
          const std::vector<size_t>& focal_length_idxs =
              camera.FocalLengthIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     focal_length_idxs.begin(),
                                     focal_length_idxs.end());
        }

        if (!options.refine_extra_params) {
          const std::vector<size_t>& extra_params_idxs =
              camera.ExtraParamsIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     extra_params_idxs.begin(),
                                     extra_params_idxs.end());
        }

        if (camera_params_const.size() == camera.NumParams()) {
          problem.SetParameterBlockConstant(camera.ParamsData());
        } else {
          SetSubsetManifold(static_cast<int>(camera.NumParams()),
                            camera_params_const, &problem, camera.ParamsData());
        }
      }
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;

  // The overhead of creating threads is too large.
  solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options.print_summary) {
    PrintHeading2("Pose refinement report");
    PrintSolverSummary(summary);
  }

  if (problem.NumResiduals() > 0 && rot_tvec_covariance != nullptr) {
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);
    std::vector<const double*> parameter_blocks = {qvec_data, tvec_data};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      return false;
    }
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 rot_tvec_covariance->data());
  }

  return summary.IsSolutionUsable();
}

}  // namespace colmap
