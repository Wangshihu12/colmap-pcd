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

#include "controllers/automatic_reconstruction.h"

#include "base/undistortion.h"
#include "controllers/incremental_mapper.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "mvs/fusion.h"
#include "mvs/meshing.h"
#include "mvs/patch_match.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

/**
 * [功能描述]：自动重建控制器的构造函数，负责初始化所有重建流程所需的配置和组件
 * @param options：重建选项配置，包含工作路径、数据类型、质量等参数
 * @param reconstruction_manager：重建管理器指针，用于管理重建结果
 */
AutomaticReconstructionController::AutomaticReconstructionController(
  const Options& options, ReconstructionManager* reconstruction_manager)
  : options_(options),
    reconstruction_manager_(reconstruction_manager),
    active_thread_(nullptr) {

// 验证必要的路径和参数是否有效
CHECK(ExistsDir(options_.workspace_path));  // 检查工作空间路径是否存在
CHECK(ExistsDir(options_.image_path));      // 检查图像路径是否存在  
CHECK_NOTNULL(reconstruction_manager_);     // 检查重建管理器指针是否为空

// 添加所有可用的配置选项到选项管理器
option_manager_.AddAllOptions();

// 设置基本路径配置
*option_manager_.image_path = options_.image_path;  // 设置图像路径
*option_manager_.database_path =                    // 设置数据库路径
    JoinPaths(options_.workspace_path, "database.db");

// 根据数据类型调整配置参数
if (options_.data_type == DataType::VIDEO) {
  option_manager_.ModifyForVideoData();       // 视频数据优化配置
} else if (options_.data_type == DataType::INDIVIDUAL) {
  option_manager_.ModifyForIndividualData();  // 单张图像数据优化配置
} else if (options_.data_type == DataType::INTERNET) {
  option_manager_.ModifyForInternetData();    // 网络图像数据优化配置
} else {
  LOG(FATAL) << "Data type not supported";   // 不支持的数据类型
}

// 验证相机模型是否有效
CHECK(ExistsCameraModelWithName(options_.camera_model));

// 根据重建质量要求调整配置参数
if (options_.quality == Quality::LOW) {
  option_manager_.ModifyForLowQuality();      // 低质量模式配置
} else if (options_.quality == Quality::MEDIUM) {
  option_manager_.ModifyForMediumQuality();   // 中等质量模式配置
} else if (options_.quality == Quality::HIGH) {
  option_manager_.ModifyForHighQuality();     // 高质量模式配置
} else if (options_.quality == Quality::EXTREME) {
  option_manager_.ModifyForExtremeQuality();  // 极高质量模式配置
}

// 设置各个处理模块的线程数量
option_manager_.sift_extraction->num_threads = options_.num_threads;  // SIFT特征提取线程数
option_manager_.sift_matching->num_threads = options_.num_threads;    // SIFT特征匹配线程数
option_manager_.mapper->num_threads = options_.num_threads;           // 映射器线程数
option_manager_.poisson_meshing->num_threads = options_.num_threads;  // 泊松网格化线程数

// 配置图像读取器选项
ImageReaderOptions& reader_options = *option_manager_.image_reader;
reader_options.database_path = *option_manager_.database_path;  // 数据库路径
reader_options.image_path = *option_manager_.image_path;        // 图像路径

// 如果提供了掩码路径，则配置掩码相关选项
if (!options_.mask_path.empty()) {
  reader_options.mask_path = options_.mask_path;                      // 图像读取器掩码路径
  option_manager_.image_reader->mask_path = options_.mask_path;       // 图像读取器掩码路径
  option_manager_.stereo_fusion->mask_path = options_.mask_path;      // 立体融合掩码路径
}

reader_options.single_camera = options_.single_camera;  // 是否使用单一相机模型
reader_options.camera_model = options_.camera_model;    // 相机模型类型

// 配置GPU使用选项
option_manager_.sift_extraction->use_gpu = options_.use_gpu;  // SIFT特征提取是否使用GPU
option_manager_.sift_matching->use_gpu = options_.use_gpu;    // SIFT特征匹配是否使用GPU

// 配置GPU设备索引
option_manager_.sift_extraction->gpu_index = options_.gpu_index;     // SIFT特征提取GPU索引
option_manager_.sift_matching->gpu_index = options_.gpu_index;       // SIFT特征匹配GPU索引
option_manager_.patch_match_stereo->gpu_index = options_.gpu_index;  // 块匹配立体GPU索引

// 创建SIFT特征提取器实例
feature_extractor_ = std::make_unique<SiftFeatureExtractor>(
    reader_options, *option_manager_.sift_extraction);

// 创建穷举特征匹配器实例（适用于小规模图像集）
exhaustive_matcher_ = std::make_unique<ExhaustiveFeatureMatcher>(
    *option_manager_.exhaustive_matching, *option_manager_.sift_matching,
    *option_manager_.database_path);

// 如果提供了词汇树路径，则启用循环检测功能
if (!options_.vocab_tree_path.empty()) {
  option_manager_.sequential_matching->loop_detection = true;        // 启用循环检测
  option_manager_.sequential_matching->vocab_tree_path =             // 设置词汇树路径
      options_.vocab_tree_path;
}

// 创建序列特征匹配器实例（适用于视频或有序图像）
sequential_matcher_ = std::make_unique<SequentialFeatureMatcher>(
    *option_manager_.sequential_matching, *option_manager_.sift_matching,
    *option_manager_.database_path);

// 如果提供了词汇树路径，则创建基于词汇树的特征匹配器
if (!options_.vocab_tree_path.empty()) {
  option_manager_.vocab_tree_matching->vocab_tree_path =  // 设置词汇树路径
      options_.vocab_tree_path;
  // 创建词汇树特征匹配器实例（适用于大规模图像集）
  vocab_tree_matcher_ = std::make_unique<VocabTreeFeatureMatcher>(
      *option_manager_.vocab_tree_matching, *option_manager_.sift_matching,
      *option_manager_.database_path);
}
}

void AutomaticReconstructionController::Stop() {
  if (active_thread_ != nullptr) {
    active_thread_->Stop();
  }
  Thread::Stop();
}

void AutomaticReconstructionController::Run() {
  if (IsStopped()) {
    return;
  }

  RunFeatureExtraction();

  if (IsStopped()) {
    return;
  }

  RunFeatureMatching();

  if (IsStopped()) {
    return;
  }

  if (options_.sparse) {
    RunSparseMapper();
  }

  if (IsStopped()) {
    return;
  }

  if (options_.dense) {
    RunDenseMapper();
  }
}

void AutomaticReconstructionController::RunFeatureExtraction() {
  CHECK(feature_extractor_);
  active_thread_ = feature_extractor_.get();
  feature_extractor_->Start();
  feature_extractor_->Wait();
  feature_extractor_.reset();
  active_thread_ = nullptr;
}

void AutomaticReconstructionController::RunFeatureMatching() {
  Thread* matcher = nullptr;
  if (options_.data_type == DataType::VIDEO) {
    matcher = sequential_matcher_.get();
  } else if (options_.data_type == DataType::INDIVIDUAL ||
             options_.data_type == DataType::INTERNET) {
    Database database(*option_manager_.database_path);
    const size_t num_images = database.NumImages();
    if (options_.vocab_tree_path.empty() || num_images < 200) {
      matcher = exhaustive_matcher_.get();
    } else {
      matcher = vocab_tree_matcher_.get();
    }
  }

  CHECK(matcher);
  active_thread_ = matcher;
  matcher->Start();
  matcher->Wait();
  exhaustive_matcher_.reset();
  sequential_matcher_.reset();
  vocab_tree_matcher_.reset();
  active_thread_ = nullptr;
}

/**
 * [功能描述]：运行稀疏映射器进行稀疏重建（SfM），这是3D重建流程中的核心步骤
 * 该函数会先检查是否已存在稀疏重建结果，如果存在则直接加载，否则执行新的稀疏重建
 * @return 无返回值
 */
void AutomaticReconstructionController::RunSparseMapper() {
  // 构建稀疏重建结果的存储路径
  const auto sparse_path = JoinPaths(options_.workspace_path, "sparse");
  
  // 检查稀疏重建目录是否已存在
  if (ExistsDir(sparse_path)) {
    // 获取sparse目录下的所有子目录列表
    auto dir_list = GetDirList(sparse_path);
    // 对目录列表进行排序，确保按顺序处理
    std::sort(dir_list.begin(), dir_list.end());
    
    // 如果存在子目录，说明已有稀疏重建结果
    if (dir_list.size() > 0) {
      // 输出警告信息，告知用户跳过稀疏重建
      std::cout << std::endl
                << "WARNING: Skipping sparse reconstruction because it is "
                   "already computed"
                << std::endl;
      
      // 遍历所有重建结果目录，加载已有的稀疏重建数据
      for (const auto& dir : dir_list) {
        reconstruction_manager_->Read(dir);
      }
      return; // 直接返回，跳过后续的重建过程
    }
  }

  // 创建增量映射器控制器，用于执行稀疏重建
  // 传入映射器配置、图像路径、数据库路径和重建管理器
  IncrementalMapperController mapper(
      option_manager_.mapper.get(), *option_manager_.image_path,
      *option_manager_.database_path, reconstruction_manager_);
  
  // 设置当前活动线程为映射器（用于线程管理和停止控制）
  active_thread_ = &mapper;
  
  // 启动增量映射器，开始稀疏重建过程
  mapper.Start();
  
  // 等待映射器完成工作
  mapper.Wait();
  
  // 清除活动线程引用
  active_thread_ = nullptr;

  // 确保sparse输出目录存在
  CreateDirIfNotExists(sparse_path);
  
  // 将重建结果写入到sparse目录中
  reconstruction_manager_->Write(sparse_path, &option_manager_);
}

void AutomaticReconstructionController::RunDenseMapper() {
  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "dense"));

  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    if (IsStopped()) {
      return;
    }

    const std::string dense_path =
        JoinPaths(options_.workspace_path, "dense", std::to_string(i));
    const std::string fused_path = JoinPaths(dense_path, "fused.ply");

    std::string meshing_path;
    if (options_.mesher == Mesher::POISSON) {
      meshing_path = JoinPaths(dense_path, "meshed-poisson.ply");
    } else if (options_.mesher == Mesher::DELAUNAY) {
      meshing_path = JoinPaths(dense_path, "meshed-delaunay.ply");
    }

    if (ExistsFile(fused_path) && ExistsFile(meshing_path)) {
      continue;
    }

    // Image undistortion.

    if (!ExistsDir(dense_path)) {
      CreateDirIfNotExists(dense_path);

      UndistortCameraOptions undistortion_options;
      undistortion_options.max_image_size =
          option_manager_.patch_match_stereo->max_image_size;
      COLMAPUndistorter undistorter(undistortion_options,
                                    reconstruction_manager_->Get(i),
                                    *option_manager_.image_path, dense_path);
      active_thread_ = &undistorter;
      undistorter.Start();
      undistorter.Wait();
      active_thread_ = nullptr;
    }

    if (IsStopped()) {
      return;
    }

    // Patch match stereo.

#ifdef CUDA_ENABLED
    {
      mvs::PatchMatchController patch_match_controller(
          *option_manager_.patch_match_stereo, dense_path, "COLMAP", "");
      active_thread_ = &patch_match_controller;
      patch_match_controller.Start();
      patch_match_controller.Wait();
      active_thread_ = nullptr;
    }
#else   // CUDA_ENABLED
    std::cout
        << std::endl
        << "WARNING: Skipping patch match stereo because CUDA is not available."
        << std::endl;
    return;
#endif  // CUDA_ENABLED

    if (IsStopped()) {
      return;
    }

    // Stereo fusion.

    if (!ExistsFile(fused_path)) {
      auto fusion_options = *option_manager_.stereo_fusion;
      const int num_reg_images = reconstruction_manager_->Get(i).NumRegImages();
      fusion_options.min_num_pixels =
          std::min(num_reg_images + 1, fusion_options.min_num_pixels);
      mvs::StereoFusion fuser(
          fusion_options, dense_path, "COLMAP", "",
          options_.quality == Quality::HIGH ? "geometric" : "photometric");
      active_thread_ = &fuser;
      fuser.Start();
      fuser.Wait();
      active_thread_ = nullptr;

      std::cout << "Writing output: " << fused_path << std::endl;
      WriteBinaryPlyPoints(fused_path, fuser.GetFusedPoints());
      mvs::WritePointsVisibility(fused_path + ".vis",
                                 fuser.GetFusedPointsVisibility());
    }

    if (IsStopped()) {
      return;
    }

    // Surface meshing.

    if (!ExistsFile(meshing_path)) {
      if (options_.mesher == Mesher::POISSON) {
        mvs::PoissonMeshing(*option_manager_.poisson_meshing, fused_path,
                            meshing_path);
      } else if (options_.mesher == Mesher::DELAUNAY) {
#ifdef CGAL_ENABLED
        mvs::DenseDelaunayMeshing(*option_manager_.delaunay_meshing, dense_path,
                                  meshing_path);
#else  // CGAL_ENABLED
        std::cout << std::endl
                  << "WARNING: Skipping Delaunay meshing because CGAL is "
                     "not available."
                  << std::endl;
        return;

#endif  // CGAL_ENABLED
      }
    }
  }
}

}  // namespace colmap
