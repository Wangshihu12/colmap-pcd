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

#ifndef COLMAP_SRC_BASE_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {
/**
 * 束调整成本函数模板类
 * 
 * 该类定义了用于Bundle Adjustment优化的成本函数，计算3D点重投影到图像平面后
 * 与实际观测2D点之间的重投影误差。这是SfM重建中最核心的优化目标函数。
 * 
 * @tparam CameraModel 相机模型类型，如PinholeCameraModel、SimplePinholeCameraModel等
 */
template <typename CameraModel>
class BundleAdjustmentCostFunction {
 public:
  /**
   * 构造函数
   * 
   * @param point2D 观测到的2D特征点坐标（像素坐标系）
   */
  explicit BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  /**
   * 创建Ceres成本函数对象的静态工厂方法
   * 
   * 该方法创建一个AutoDiffCostFunction对象，用于Ceres Solver的自动微分优化。
   * 模板参数说明：
   * - BundleAdjustmentCostFunction<CameraModel>: 成本函数类型
   * - 2: 残差维度（重投影误差的x和y分量）
   * - 4: 第一个参数块维度（四元数旋转，4个参数）
   * - 3: 第二个参数块维度（平移向量，3个参数）  
   * - 3: 第三个参数块维度（3D点坐标，3个参数）
   * - CameraModel::kNumParams: 第四个参数块维度（相机内参个数，取决于相机模型）
   * 
   * @param point2D 观测到的2D特征点坐标
   * @return 指向Ceres成本函数对象的指针
   */
  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentCostFunction<CameraModel>, 
                2,    // 残差维度：重投影误差的x,y分量
                4,    // 旋转四元数参数维度
                3,    // 平移向量参数维度  
                3,    // 3D点坐标参数维度
                CameraModel::kNumParams  // 相机内参参数维度
            >(new BundleAdjustmentCostFunction(point2D)));
  }

  /**
   * 成本函数的核心计算逻辑（函数调用运算符重载）
   * 
   * 该函数实现了从3D点到2D图像点的完整投影流程，并计算重投影误差。
   * 投影流程：世界坐标 -> 相机坐标 -> 归一化坐标 -> 像素坐标
   * 
   * @tparam T 数值类型，支持双精度double和Ceres的Jet类型（用于自动微分）
   * @param qvec 相机旋转四元数 [w, x, y, z]，表示世界到相机的旋转
   * @param tvec 相机平移向量 [tx, ty, tz]，表示世界到相机的平移
   * @param point3D 3D点的世界坐标 [X, Y, Z]
   * @param camera_params 相机内参数组，具体参数取决于相机模型
   * @param residuals 输出的残差数组 [rx, ry]，即重投影误差
   * @return 总是返回true，表示计算成功
   */
  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    
    // 步骤1：世界坐标系到相机坐标系的坐标变换
    // 首先使用四元数将3D点从世界坐标系旋转到相机坐标系
    T projection[3];  // 临时存储相机坐标系下的3D点坐标
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    
    // 然后加上平移分量，完成完整的刚体变换
    // P_camera = R * P_world + t
    projection[0] += tvec[0];  // X坐标分量
    projection[1] += tvec[1];  // Y坐标分量  
    projection[2] += tvec[2];  // Z坐标分量（深度）

    // 步骤2：透视投影 - 将3D点投影到归一化图像平面
    // 执行透视除法：(X/Z, Y/Z)，得到归一化坐标
    // 这里假设Z > 0（点在相机前方）
    projection[0] /= projection[2];  // 归一化x坐标
    projection[1] /= projection[2];  // 归一化y坐标

    // 步骤3：相机畸变校正和像素坐标转换
    // 使用特定的相机模型将归一化坐标转换为像素坐标
    // 这一步会考虑相机内参（焦距、主光轴偏移）和畸变参数（径向畸变、切向畸变等）
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // 步骤4：计算重投影误差
    // 重投影误差 = 预测像素坐标 - 观测像素坐标
    // 这是Bundle Adjustment要最小化的目标函数
    residuals[0] -= T(observed_x_);  // x方向重投影误差
    residuals[1] -= T(observed_y_);  // y方向重投影误差

    return true;  // 表示计算成功完成
  }

 private:
  // 存储观测到的2D特征点坐标（像素坐标系）
  const double observed_x_;  // 观测点的x坐标
  const double observed_y_;  // 观测点的y坐标
};

/**
 * 激光雷达束调整成本函数类
 * 
 * 该类定义了在Bundle Adjustment优化中融入激光雷达点云约束的成本函数。
 * 通过计算3D点到激光雷达观测平面的距离来构建约束，实现视觉-激光雷达融合优化。
 * 这种约束有助于提高SfM重建的几何精度和尺度准确性。
 */
class BundleAdjustmentLidarCostFunction {
 public: 
  /**
   * 构造函数
   * 
   * @param abcd 平面方程系数向量 [a, b, c, d]，表示平面方程 ax + by + cz + d = 0
   *             其中 (a, b, c) 是平面法向量，d 是平面到原点的距离参数
   * @param weight 权重因子，用于控制激光雷达约束相对于视觉重投影约束的重要性
   *               权重越大，激光雷达约束影响越强
   */
  explicit BundleAdjustmentLidarCostFunction(const Eigen::Matrix<double,4,1>& abcd, const double& weight)
      : weight_(weight),           // 存储权重系数
        a_(abcd(0)),              // 平面方程中的a系数（法向量x分量）
        b_(abcd(1)),              // 平面方程中的b系数（法向量y分量）  
        c_(abcd(2)),              // 平面方程中的c系数（法向量z分量）
        d_(abcd(3)) {}            // 平面方程中的d系数（距离参数）

  /**
   * 创建Ceres成本函数对象的静态工厂方法
   * 
   * 该方法创建一个AutoDiffCostFunction对象，用于Ceres Solver的自动微分优化。
   * 模板参数说明：
   * - BundleAdjustmentLidarCostFunction: 成本函数类型
   * - 1: 残差维度（点到平面距离，标量值）
   * - 3: 参数块维度（3D点坐标，3个参数：x, y, z）
   * 
   * @param abcd 平面方程系数向量
   * @param weight 权重因子
   * @return 指向Ceres成本函数对象的指针
   */
  static ceres::CostFunction* Create(const Eigen::Matrix<double,4,1>& abcd, const double& weight){
    return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarCostFunction,
                1,    // 残差维度：点到平面距离（标量）
                3     // 3D点坐标参数维度：[x, y, z]
            >(new BundleAdjustmentLidarCostFunction(abcd, weight)));
  }

  /**
   * 成本函数的核心计算逻辑（函数调用运算符重载）
   * 
   * 该函数计算3D点到激光雷达观测平面的加权距离作为残差。
   * 这个距离越小，说明3D点越接近激光雷达观测到的真实表面。
   * 
   * 数学原理：
   * - 点到平面距离公式：|ax + by + cz + d| / sqrt(a² + b² + c²)
   * - 这里假设法向量已经归一化，即 sqrt(a² + b² + c²) = 1
   * - 因此距离简化为：|ax + by + cz + d|
   * 
   * @tparam T 数值类型，支持双精度double和Ceres的Jet类型（用于自动微分）
   * @param point3D 待优化的3D点坐标 [x, y, z]
   * @param residuals 输出的残差数组，包含一个元素：加权的点到平面距离
   * @return 总是返回true，表示计算成功
   */
  template <typename T>
  bool operator()(const T* const point3D, T* residuals) const {
    // 计算加权的点到平面距离作为残差
    // 公式分解：
    // 1. point3D[0] * a_ + point3D[1] * b_ + point3D[2] * c_ + d_
    //    这是平面方程 ax + by + cz + d 的计算结果
    //    如果点在平面上，结果为0；否则表示点到平面的有符号距离
    // 
    // 2. T(0.) - (ax + by + cz + d)
    //    将有符号距离取负，目标是让这个值趋近于0
    //    即让点尽可能接近平面
    // 
    // 3. 平方操作：(...) * (...)
    //    将有符号距离转换为无符号的平方距离，避免符号问题
    //    同时增强对较大偏差的惩罚
    // 
    // 4. sqrt(平方距离)
    //    取平方根得到欧几里得距离，这是更自然的几何意义
    // 
    // 5. T(weight_) * 距离
    //    应用权重因子，控制激光雷达约束的相对重要性
    
    residuals[0] = T(weight_) * ceres::sqrt(
        (T(0.) - (point3D[0] * a_ + point3D[1] * b_ + point3D[2] * c_ + d_)) *
        (T(0.) - (point3D[0] * a_ + point3D[1] * b_ + point3D[2] * c_ + d_))
    );
    
    return true;  // 表示计算成功完成
  }

 private:
  // 存储激光雷达约束参数
  const double weight_;  // 权重因子，控制约束强度
  const double a_;       // 平面方程系数a（法向量x分量）
  const double b_;       // 平面方程系数b（法向量y分量）
  const double c_;       // 平面方程系数c（法向量z分量）
  const double d_;       // 平面方程系数d（距离参数）
};

/**
 * 固定位姿的束调整成本函数模板类
 * 
 * 该类是Bundle Adjustment中的一个特殊变体，用于处理某些相机位姿需要保持固定不变的情况。
 * 与标准束调整不同，这里相机的旋转和平移参数被固定为常量，只优化3D点坐标和相机内参。
 * 
 * 应用场景：
 * - 参考图像的位姿需要固定以避免整体漂移
 * - 某些图像的位姿已知且需要保持不变
 * - 分阶段优化中需要固定部分相机参数
 * 
 * @tparam CameraModel 相机模型类型，如PinholeCameraModel、SimplePinholeCameraModel等
 */
template <typename CameraModel>
class BundleAdjustmentConstantPoseCostFunction {
 public:
  /**
   * 构造函数
   * 
   * @param qvec 固定的相机旋转四元数 [w, x, y, z]，表示世界到相机的旋转
   * @param tvec 固定的相机平移向量 [tx, ty, tz]，表示世界到相机的平移
   * @param point2D 观测到的2D特征点坐标（像素坐标系）
   */
  BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec,
                                           const Eigen::Vector2d& point2D)
      : qw_(qvec(0)),                    // 四元数的w分量（实部）
        qx_(qvec(1)),                    // 四元数的x分量（虚部i）
        qy_(qvec(2)),                    // 四元数的y分量（虚部j）
        qz_(qvec(3)),                    // 四元数的z分量（虚部k）
        tx_(tvec(0)),                    // 平移向量的x分量
        ty_(tvec(1)),                    // 平移向量的y分量
        tz_(tvec(2)),                    // 平移向量的z分量
        observed_x_(point2D(0)),         // 观测点的x坐标
        observed_y_(point2D(1)) {}       // 观测点的y坐标

  /**
   * 创建Ceres成本函数对象的静态工厂方法
   * 
   * 该方法创建一个AutoDiffCostFunction对象，用于Ceres Solver的自动微分优化。
   * 注意：与标准束调整相比，这里没有位姿参数块，因为位姿被固定为常量。
   * 
   * 模板参数说明：
   * - BundleAdjustmentConstantPoseCostFunction<CameraModel>: 成本函数类型
   * - 2: 残差维度（重投影误差的x和y分量）
   * - 3: 第一个参数块维度（3D点坐标，3个参数）
   * - CameraModel::kNumParams: 第二个参数块维度（相机内参个数）
   * 
   * @param qvec 固定的相机旋转四元数
   * @param tvec 固定的相机平移向量
   * @param point2D 观测到的2D特征点坐标
   * @return 指向Ceres成本函数对象的指针
   */
  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentConstantPoseCostFunction<CameraModel>, 
            2,                           // 残差维度：重投影误差的x,y分量
            3,                           // 3D点坐标参数维度
            CameraModel::kNumParams      // 相机内参参数维度
        >(new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D)));
  }

  /**
   * 成本函数的核心计算逻辑（函数调用运算符重载）
   * 
   * 该函数使用固定的相机位姿计算3D点的重投影误差。
   * 与标准束调整的区别是位姿参数来自类的成员变量而非优化参数。
   * 
   * @tparam T 数值类型，支持双精度double和Ceres的Jet类型（用于自动微分）
   * @param point3D 待优化的3D点坐标 [X, Y, Z]
   * @param camera_params 待优化的相机内参数组
   * @param residuals 输出的残差数组 [rx, ry]，即重投影误差
   * @return 总是返回true，表示计算成功
   */
  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    
    // 将存储的固定位姿参数转换为模板类型T
    // 这样做是为了支持Ceres的自动微分机制
    const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

    // 步骤1：世界坐标系到相机坐标系的坐标变换
    // 使用固定的四元数将3D点从世界坐标系旋转到相机坐标系
    T projection[3];  // 临时存储相机坐标系下的3D点坐标
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    
    // 加上固定的平移分量，完成完整的刚体变换
    // P_camera = R_fixed * P_world + t_fixed
    projection[0] += T(tx_);  // X坐标分量
    projection[1] += T(ty_);  // Y坐标分量
    projection[2] += T(tz_);  // Z坐标分量（深度）

    // 步骤2：透视投影 - 将3D点投影到归一化图像平面
    // 执行透视除法：(X/Z, Y/Z)，得到归一化坐标
    projection[0] /= projection[2];  // 归一化x坐标
    projection[1] /= projection[2];  // 归一化y坐标

    // 步骤3：相机畸变校正和像素坐标转换
    // 使用可优化的相机内参将归一化坐标转换为像素坐标
    // 注意：这里相机内参是可以被优化的，与固定的位姿形成对比
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // 步骤4：计算重投影误差
    // 重投影误差 = 预测像素坐标 - 观测像素坐标
    residuals[0] -= T(observed_x_);  // x方向重投影误差
    residuals[1] -= T(observed_y_);  // y方向重投影误差

    return true;  // 表示计算成功完成
  }

 private:
  // 存储固定的相机位姿参数（这些参数在优化过程中保持不变）
  const double qw_;           // 旋转四元数的w分量
  const double qx_;           // 旋转四元数的x分量
  const double qy_;           // 旋转四元数的y分量
  const double qz_;           // 旋转四元数的z分量
  const double tx_;           // 平移向量的x分量
  const double ty_;           // 平移向量的y分量
  const double tz_;           // 平移向量的z分量
  
  // 存储观测到的2D特征点坐标
  const double observed_x_;   // 观测点的x坐标
  const double observed_y_;   // 观测点的y坐标
};

/**
 * 固定3D点的束调整成本函数模板类
 * 
 * 该类是Bundle Adjustment中的另一个特殊变体，用于处理3D点坐标已知且需要保持固定的情况。
 * 与标准束调整不同，这里3D点坐标被固定为常量，只优化相机位姿和相机内参。
 * 
 * 应用场景：
 * - 使用已知的控制点进行相机标定
 * - 3D点坐标通过其他高精度方法（如激光雷达）获得
 * - 分阶段优化中需要固定3D结构
 * - 相机位姿估计和内参标定
 * 
 * @tparam CameraModel 相机模型类型，如PinholeCameraModel、SimplePinholeCameraModel等
 */
template <typename CameraModel>
class BundleAdjustmentConstantPoint3DCostFunction {
 public:
  /**
   * 构造函数
   * 
   * @param point2D 观测到的2D特征点坐标（像素坐标系）
   * @param point3D 固定的3D点世界坐标 [X, Y, Z]
   */
  explicit BundleAdjustmentConstantPoint3DCostFunction(
      const Eigen::Vector2d& point2D, const Eigen::Vector3d& point3D)
      : observed_x_(point2D(0)),        // 观测点的x坐标
        observed_y_(point2D(1)),        // 观测点的y坐标
        point3D_x_(point3D(0)),         // 固定3D点的X坐标
        point3D_y_(point3D(1)),         // 固定3D点的Y坐标
        point3D_z_(point3D(2)) {}       // 固定3D点的Z坐标

  /**
   * 创建Ceres成本函数对象的静态工厂方法
   * 
   * 该方法创建一个AutoDiffCostFunction对象，用于Ceres Solver的自动微分优化。
   * 注意：与标准束调整相比，这里没有3D点坐标参数块，因为3D点被固定为常量。
   * 
   * 模板参数说明：
   * - BundleAdjustmentConstantPoint3DCostFunction<CameraModel>: 成本函数类型
   * - 2: 残差维度（重投影误差的x和y分量）
   * - 4: 第一个参数块维度（四元数旋转，4个参数）
   * - 3: 第二个参数块维度（平移向量，3个参数）
   * - CameraModel::kNumParams: 第三个参数块维度（相机内参个数）
   * 
   * @param point2D 观测到的2D特征点坐标
   * @param point3D 固定的3D点世界坐标
   * @return 指向Ceres成本函数对象的指针
   */
  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D,
                                     const Eigen::Vector3d& point3D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentConstantPoint3DCostFunction<CameraModel>, 
            2,                           // 残差维度：重投影误差的x,y分量
            4,                           // 旋转四元数参数维度
            3,                           // 平移向量参数维度
            CameraModel::kNumParams      // 相机内参参数维度
        >(new BundleAdjustmentConstantPoint3DCostFunction(point2D, point3D)));
  }

  /**
   * 成本函数的核心计算逻辑（函数调用运算符重载）
   * 
   * 该函数使用固定的3D点坐标和可优化的相机参数计算重投影误差。
   * 与标准束调整的区别是3D点坐标来自类的成员变量而非优化参数。
   * 
   * @tparam T 数值类型，支持双精度double和Ceres的Jet类型（用于自动微分）
   * @param qvec 待优化的相机旋转四元数 [w, x, y, z]
   * @param tvec 待优化的相机平移向量 [tx, ty, tz]
   * @param camera_params 待优化的相机内参数组
   * @param residuals 输出的残差数组 [rx, ry]，即重投影误差
   * @return 总是返回true，表示计算成功
   */
  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const camera_params, T* residuals) const {
    
    // 将存储的固定3D点坐标转换为模板类型T
    // 这样做是为了支持Ceres的自动微分机制
    // 注意：这些3D点坐标在优化过程中保持不变
    const T point3D[3] = {T(point3D_x_), T(point3D_y_), T(point3D_z_)};

    // 步骤1：世界坐标系到相机坐标系的坐标变换
    // 使用可优化的四元数将固定的3D点从世界坐标系旋转到相机坐标系
    T projection[3];  // 临时存储相机坐标系下的3D点坐标
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    
    // 加上可优化的平移分量，完成完整的刚体变换
    // P_camera = R_optimizable * P_world_fixed + t_optimizable
    projection[0] += tvec[0];  // X坐标分量
    projection[1] += tvec[1];  // Y坐标分量
    projection[2] += tvec[2];  // Z坐标分量（深度）

    // 步骤2：透视投影 - 将3D点投影到归一化图像平面
    // 执行透视除法：(X/Z, Y/Z)，得到归一化坐标
    projection[0] /= projection[2];  // 归一化x坐标
    projection[1] /= projection[2];  // 归一化y坐标

    // 步骤3：相机畸变校正和像素坐标转换
    // 使用可优化的相机内参将归一化坐标转换为像素坐标
    // 这里相机内参和位姿都是可以被优化的，只有3D点坐标是固定的
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // 步骤4：计算重投影误差
    // 重投影误差 = 预测像素坐标 - 观测像素坐标
    residuals[0] -= T(observed_x_);  // x方向重投影误差
    residuals[1] -= T(observed_y_);  // y方向重投影误差

    return true;  // 表示计算成功完成
  }

 private:
  // 存储观测到的2D特征点坐标
  const double observed_x_;   // 观测点的x坐标
  const double observed_y_;   // 观测点的y坐标
  
  // 存储固定的3D点世界坐标（这些坐标在优化过程中保持不变）
  const double point3D_x_;    // 固定3D点的X坐标
  const double point3D_y_;    // 固定3D点的Y坐标
  const double point3D_z_;    // 固定3D点的Z坐标
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigBundleAdjustmentCostFunction {
 public:
  explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            RigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
            CameraModel::kNumParams>(
        new RigBundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const rig_qvec, const T* const rig_tvec,
                  const T* const rel_qvec, const T* const rel_tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // Concatenate rotations.
    T qvec[4];
    ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

    // Concatenate translations.
    T tvec[3];
    ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
    tvec[0] += rel_tvec[0];
    tvec[1] += rel_tvec[1];
    tvec[2] += rel_tvec[2];

    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += tvec[0];
    projection[1] += tvec[1];
    projection[2] += tvec[2];

    // Project to image plane.
    projection[0] /= projection[2];
    projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);

    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `SphereManifold`.
class RelativePoseCostFunction {
 public:
  RelativePoseCostFunction(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) {
    return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(
        new RelativePoseCostFunction(x1, x2)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
    ceres::QuaternionToRotation(qvec, R.data());

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0],
        T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    // Homogeneous image coordinates.
    const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
    const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
    const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
    const T x2tEx1 = x2_h.transpose() * Ex1;
    residuals[0] = x2tEx1 * x2tEx1 /
                   (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                    Etx2(1) * Etx2(1));

    return true;
  }

 private:
  const double x1_;
  const double y1_;
  const double x2_;
  const double y2_;
};

inline void SetQuaternionManifold(ceres::Problem* problem, double* qvec) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
  problem->SetManifold(qvec, new ceres::QuaternionManifold);
#else
  problem->SetParameterization(qvec, new ceres::QuaternionParameterization);
#endif
}

inline void SetSubsetManifold(int size, const std::vector<int>& constant_params,
                              ceres::Problem* problem, double* params) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
  problem->SetManifold(params,
                       new ceres::SubsetManifold(size, constant_params));
#else
  problem->SetParameterization(
      params, new ceres::SubsetParameterization(size, constant_params));
#endif
}

template <int size>
inline void SetSphereManifold(ceres::Problem* problem, double* params) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
  problem->SetManifold(params, new ceres::SphereManifold<size>);
#else
  problem->SetParameterization(
      params, new ceres::HomogeneousVectorParameterization(size));
#endif
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_COST_FUNCTIONS_H_
