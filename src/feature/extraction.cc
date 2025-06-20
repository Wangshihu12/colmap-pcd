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

#include "feature/extraction.h"

#include <numeric>

#include "SiftGPU/SiftGPU.h"
#include "feature/sift.h"
#include "util/cuda.h"
#include "util/misc.h"

namespace colmap {
namespace {

void ScaleKeypoints(const Bitmap& bitmap, const Camera& camera,
                    FeatureKeypoints* keypoints) {
  if (static_cast<size_t>(bitmap.Width()) != camera.Width() ||
      static_cast<size_t>(bitmap.Height()) != camera.Height()) {
    const float scale_x = static_cast<float>(camera.Width()) / bitmap.Width();
    const float scale_y = static_cast<float>(camera.Height()) / bitmap.Height();
    for (auto& keypoint : *keypoints) {
      keypoint.Rescale(scale_x, scale_y);
    }
  }
}

void MaskKeypoints(const Bitmap& mask, FeatureKeypoints* keypoints,
                   FeatureDescriptors* descriptors) {
  size_t out_index = 0;
  BitmapColor<uint8_t> color;
  for (size_t i = 0; i < keypoints->size(); ++i) {
    if (!mask.GetPixel(static_cast<int>(keypoints->at(i).x),
                       static_cast<int>(keypoints->at(i).y), &color) ||
        color.r == 0) {
      // Delete this keypoint by not copying it to the output.
    } else {
      // Retain this keypoint by copying it to the output index (in case this
      // index differs from its current position).
      if (out_index != i) {
        keypoints->at(out_index) = keypoints->at(i);
        for (int col = 0; col < descriptors->cols(); ++col) {
          (*descriptors)(out_index, col) = (*descriptors)(i, col);
        }
      }
      out_index += 1;
    }
  }

  keypoints->resize(out_index);
  descriptors->conservativeResize(out_index, descriptors->cols());
}

}  // namespace

/**
 * SIFT特征提取器构造函数
 * 
 * 该构造函数初始化特征提取的多线程处理管道，包括图像读取、调整大小、特征提取和结果写入。
 * 支持CPU和GPU两种提取模式，并可根据可用硬件资源自动配置并行处理。
 * 
 * @param reader_options 图像读取配置选项，包含图像路径、数据库路径等
 * @param sift_options SIFT特征提取的具体参数配置
 */
SiftFeatureExtractor::SiftFeatureExtractor(
    const ImageReaderOptions& reader_options,
    const SiftExtractionOptions& sift_options)
    : reader_options_(reader_options),
      sift_options_(sift_options),
      database_(reader_options_.database_path),//database_使用database_path的string显式初始化，打开reader_options_.database_path文件
      //image_reader_初始化
      image_reader_(reader_options_, &database_) { // 初始化图像读取器

  // 检查输入参数的有效性
  CHECK(reader_options_.Check());
  CHECK(sift_options_.Check());

  // 处理相机掩码（如果指定）
  // 相机掩码用于排除图像中不需要提取特征的区域，如镜头污点、水印等
  std::shared_ptr<Bitmap> camera_mask;
  if (!reader_options_.camera_mask_path.empty()) {
    camera_mask = std::make_shared<Bitmap>();
    if (!camera_mask->Read(reader_options_.camera_mask_path,
                           /*as_rgb*/ false)) {
      std::cerr << "  ERROR: Cannot read camera mask file: "
                << reader_options_.camera_mask_path
                << ". No mask is going to be used." << std::endl;
      camera_mask.reset(); // 读取失败时放弃使用掩码
    }
  }

  // 确定实际使用的线程数
  // GetEffectiveNumThreads会根据请求的线程数和系统可用核心数选择合适的值
  const int num_threads = GetEffectiveNumThreads(sift_options_.num_threads);
  CHECK_GT(num_threads, 0); // 确保至少有一个线程

  // 创建处理管道的任务队列
  // 限制队列大小为1，避免过多图像同时加载到内存造成内存占用过高
  const int kQueueSize = 1;
  // 创建三个处理阶段的任务队列：调整大小、特征提取和结果写入
  resizer_queue_ = std::make_unique<JobQueue<internal::ImageData>>(kQueueSize);
  extractor_queue_ =
      std::make_unique<JobQueue<internal::ImageData>>(kQueueSize);
  writer_queue_ = std::make_unique<JobQueue<internal::ImageData>>(kQueueSize);

  // 如果需要调整图像大小（限制最大尺寸），创建图像调整线程
  if (sift_options_.max_image_size > 0) {
    for (int i = 0; i < num_threads; ++i) {
      resizers_.emplace_back(std::make_unique<internal::ImageResizerThread>(
          sift_options_.max_image_size, resizer_queue_.get(),
          extractor_queue_.get()));
    }
  }

  // 根据SIFT参数和GPU可用性选择不同的特征提取方法
  // 当不需要域大小池化、不估计仿射形状且启用GPU时，使用GPU特征提取
  if (!sift_options_.domain_size_pooling &&
      !sift_options_.estimate_affine_shape && sift_options_.use_gpu) {

    // 解析GPU索引（可以是逗号分隔的多个索引）
    std::vector<int> gpu_indices = CSVToVector<int>(sift_options_.gpu_index);
    CHECK_GT(gpu_indices.size(), 0); // 确保至少指定了一个GPU

#ifdef CUDA_ENABLED
    // 如果GPU索引为-1，表示使用所有可用的CUDA设备
    if (gpu_indices.size() == 1 && gpu_indices[0] == -1) {
      const int num_cuda_devices = GetNumCudaDevices();
      CHECK_GT(num_cuda_devices, 0); // 确保有可用的CUDA设备
      gpu_indices.resize(num_cuda_devices);
      std::iota(gpu_indices.begin(), gpu_indices.end(), 0); // 填充0到n-1的索引
    }
#endif  // CUDA_ENABLED

    // 为每个GPU创建一个特征提取线程
    auto sift_gpu_options = sift_options_;
    for (const auto& gpu_index : gpu_indices) {
      sift_gpu_options.gpu_index = std::to_string(gpu_index);
      extractors_.emplace_back(
          std::make_unique<internal::SiftFeatureExtractorThread>(
              sift_gpu_options, camera_mask, extractor_queue_.get(),
              writer_queue_.get()));
    }
  } else {

    // 使用CPU特征提取
    // 当使用最大线程数且未调整其他参数时，显示内存使用警告
    if (sift_options_.num_threads == -1 &&
        sift_options_.max_image_size ==
            SiftExtractionOptions().max_image_size &&
        sift_options_.first_octave == SiftExtractionOptions().first_octave) {
      std::cout
          << "WARNING: Your current options use the maximum number of "
             "threads on the machine to extract features. Extracting SIFT "
             "features on the CPU can consume a lot of RAM per thread for "
             "large images. Consider reducing the maximum image size and/or "
             "the first octave or manually limit the number of extraction "
             "threads. Ignore this warning, if your machine has sufficient "
             "memory for the current settings."
          << std::endl;
    }

    // 创建CPU特征提取线程
    auto custom_sift_options = sift_options_;
    custom_sift_options.use_gpu = false; // 强制使用CPU
    for (int i = 0; i < num_threads; ++i) {
      extractors_.emplace_back(
          std::make_unique<internal::SiftFeatureExtractorThread>(
              custom_sift_options, camera_mask, extractor_queue_.get(),
              writer_queue_.get()));
    }
  }

  // 创建特征写入线程，负责将提取的特征写入数据库
  writer_ = std::make_unique<internal::FeatureWriterThread>(
      image_reader_.NumImages(), &database_, writer_queue_.get());
}

void SiftFeatureExtractor::Run() {
  PrintHeading1("Feature extraction");//打印标题

  for (auto& resizer : resizers_) {
    resizer->Start();
  }

  for (auto& extractor : extractors_) {
    extractor->Start();
  }

  writer_->Start();

  for (auto& extractor : extractors_) {
    if (!extractor->CheckValidSetup()) {
      return;
    }
  }

  while (image_reader_.NextIndex() < image_reader_.NumImages()) {
    if (IsStopped()) {
      resizer_queue_->Stop();
      extractor_queue_->Stop();
      resizer_queue_->Clear();
      extractor_queue_->Clear();
      break;
    }

    internal::ImageData image_data;
    image_data.status =
        image_reader_.Next(&image_data.camera, &image_data.image,
                           &image_data.bitmap, &image_data.mask);

    if (image_data.status != ImageReader::Status::SUCCESS) {
      image_data.bitmap.Deallocate();
    }

    if (sift_options_.max_image_size > 0) {
      CHECK(resizer_queue_->Push(std::move(image_data)));
    } else {
      CHECK(extractor_queue_->Push(std::move(image_data)));
    }
  }

  resizer_queue_->Wait();
  resizer_queue_->Stop();
  for (auto& resizer : resizers_) {
    resizer->Wait();
  }

  extractor_queue_->Wait();
  extractor_queue_->Stop();
  for (auto& extractor : extractors_) {
    extractor->Wait();
  }

  writer_queue_->Wait();
  writer_queue_->Stop();
  writer_->Wait();

  GetTimer().PrintMinutes();
}

FeatureImporter::FeatureImporter(const ImageReaderOptions& reader_options,
                                 const std::string& import_path)
    : reader_options_(reader_options), import_path_(import_path) {}

void FeatureImporter::Run() {
  PrintHeading1("Feature import");

  if (!ExistsDir(import_path_)) {
    std::cerr << "  ERROR: Import directory does not exist." << std::endl;
    return;
  }

  Database database(reader_options_.database_path);
  ImageReader image_reader(reader_options_, &database);

  while (image_reader.NextIndex() < image_reader.NumImages()) {
    if (IsStopped()) {
      break;
    }

    std::cout << StringPrintf("Processing file [%d/%d]",
                              image_reader.NextIndex() + 1,
                              image_reader.NumImages())
              << std::endl;

    // Load image data and possibly save camera to database.
    Camera camera;
    Image image;
    Bitmap bitmap;
    if (image_reader.Next(&camera, &image, &bitmap, nullptr) !=
        ImageReader::Status::SUCCESS) {
      continue;
    }

    const std::string path = JoinPaths(import_path_, image.Name() + ".txt");

    if (ExistsFile(path)) {
      FeatureKeypoints keypoints;
      FeatureDescriptors descriptors;
      LoadSiftFeaturesFromTextFile(path, &keypoints, &descriptors);

      std::cout << "  Features:       " << keypoints.size() << std::endl;

      DatabaseTransaction database_transaction(&database);

      if (image.ImageId() == kInvalidImageId) {
        image.SetImageId(database.WriteImage(image));
      }

      if (!database.ExistsKeypoints(image.ImageId())) {
        database.WriteKeypoints(image.ImageId(), keypoints);
      }

      if (!database.ExistsDescriptors(image.ImageId())) {
        database.WriteDescriptors(image.ImageId(), descriptors);
      }
    } else {
      std::cout << "  SKIP: No features found at " << path << std::endl;
    }
  }

  GetTimer().PrintMinutes();
}

namespace internal {

ImageResizerThread::ImageResizerThread(const int max_image_size,
                                       JobQueue<ImageData>* input_queue,
                                       JobQueue<ImageData>* output_queue)
    : max_image_size_(max_image_size),
      input_queue_(input_queue),
      output_queue_(output_queue) {}

void ImageResizerThread::Run() {
  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& image_data = input_job.Data();

      if (image_data.status == ImageReader::Status::SUCCESS) {
        if (static_cast<int>(image_data.bitmap.Width()) > max_image_size_ ||
            static_cast<int>(image_data.bitmap.Height()) > max_image_size_) {
          // Fit the down-sampled version exactly into the max dimensions.
          const double scale =
              static_cast<double>(max_image_size_) /
              std::max(image_data.bitmap.Width(), image_data.bitmap.Height());
          const int new_width =
              static_cast<int>(image_data.bitmap.Width() * scale);
          const int new_height =
              static_cast<int>(image_data.bitmap.Height() * scale);

          image_data.bitmap.Rescale(new_width, new_height);
        }
      }

      output_queue_->Push(std::move(image_data));
    } else {
      break;
    }
  }
}

SiftFeatureExtractorThread::SiftFeatureExtractorThread(
    const SiftExtractionOptions& sift_options,
    const std::shared_ptr<Bitmap>& camera_mask,
    JobQueue<ImageData>* input_queue, JobQueue<ImageData>* output_queue)
    : sift_options_(sift_options),
      camera_mask_(camera_mask),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(sift_options_.Check());

#ifndef CUDA_ENABLED
  if (sift_options_.use_gpu) {
    opengl_context_ = std::make_unique<OpenGLContextManager>();
  }
#endif
}

/**
 * SIFT特征提取线程的主执行函数
 * 
 * 该函数实现了特征提取线程的核心逻辑，支持多种SIFT特征提取模式：
 * 1. GPU加速的SIFT特征提取
 * 2. CPU上的标准SIFT特征提取
 * 3. 带有仿射形状估计的协变SIFT特征提取
 * 
 * 线程持续从输入队列获取图像任务，提取特征后将结果放入输出队列。
 */
void SiftFeatureExtractorThread::Run() {

  // SiftGPU对象指针，仅在使用GPU模式时初始化
  std::unique_ptr<SiftGPU> sift_gpu;

  // 如果配置为使用GPU进行特征提取
  if (sift_options_.use_gpu) {
#ifndef CUDA_ENABLED
    // 在没有CUDA支持时，需要使用OpenGL上下文
    CHECK(opengl_context_); // 确保OpenGL上下文已创建
    CHECK(opengl_context_->MakeCurrent()); // 将当前线程设为OpenGL上下文
#endif

    // 创建SiftGPU对象
    sift_gpu = std::make_unique<SiftGPU>();

    // 尝试初始化SiftGPU提取器，设置各种参数
    if (!CreateSiftGPUExtractor(sift_options_, sift_gpu.get())) {
      std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;

      // 向线程管理器报告无效设置状态，这将导致特征提取过程中止
      SignalInvalidSetup();
      return;
    }
  }

  // 向线程管理器报告设置成功，可以开始处理任务
  SignalValidSetup();

  // 主处理循环 - 持续处理任务直到收到停止信号或队列关闭
  while (true) {
    // 检查是否收到停止信号
    if (IsStopped()) {
      break;
    }

    // 从输入队列获取下一个图像任务
    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) { // 如果获取到有效任务

      // 获取图像数据引用
      auto& image_data = input_job.Data();

      // 只处理成功读取的图像
      if (image_data.status == ImageReader::Status::SUCCESS) {
        bool success = false;

        // 根据配置选择不同的特征提取方法

        // 1. 如果需要估计仿射形状或使用域大小池化，使用协变SIFT特征提取
        if (sift_options_.estimate_affine_shape ||
            sift_options_.domain_size_pooling) {
          success = ExtractCovariantSiftFeaturesCPU(
              sift_options_, image_data.bitmap, &image_data.keypoints,
              &image_data.descriptors);
        } else if (sift_options_.use_gpu) {
          // 2. 如果配置使用GPU且不需要上述特殊处理，使用GPU加速版SIFT
          success = ExtractSiftFeaturesGPU(
              sift_options_, image_data.bitmap, sift_gpu.get(),
              &image_data.keypoints, &image_data.descriptors);
        } else {
          // 3. 否则使用标准CPU版SIFT提取
          success = ExtractSiftFeaturesCPU(sift_options_, image_data.bitmap,
                                           &image_data.keypoints,
                                           &image_data.descriptors);
        }

        // 特征提取成功后的后处理
        if (success) {
          // 根据相机参数调整关键点的尺度和位置
          // 这确保了特征点在原始未调整大小的图像坐标系中的正确位置
          ScaleKeypoints(image_data.bitmap, image_data.camera,
                         &image_data.keypoints);

          // 应用全局相机掩码（如果存在）
          // 这可以排除镜头边缘或已知有问题的区域
          if (camera_mask_) {
            MaskKeypoints(*camera_mask_, &image_data.keypoints,
                          &image_data.descriptors);
          }

          // 应用图像特定掩码（如果存在）
          // 这可以排除特定图像中不需要的区域，如水印或动态物体
          if (image_data.mask.Data()) {
            MaskKeypoints(image_data.mask, &image_data.keypoints,
                          &image_data.descriptors);
          }
        } else {
          // 如果特征提取失败，将图像状态标记为失败
          image_data.status = ImageReader::Status::FAILURE;
        }
      }

      // 释放图像位图数据，减少内存占用
      // 此时特征已提取完毕，原始图像数据不再需要
      image_data.bitmap.Deallocate();

      // 将处理结果（无论成功或失败）推送到输出队列
      output_queue_->Push(std::move(image_data));
    } else {
      break;
    }
  }
}

FeatureWriterThread::FeatureWriterThread(const size_t num_images,
                                         Database* database,
                                         JobQueue<ImageData>* input_queue)
    : num_images_(num_images), database_(database), input_queue_(input_queue) {}

void FeatureWriterThread::Run() {
  size_t image_index = 0;
  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& image_data = input_job.Data();

      image_index += 1;

      std::cout << StringPrintf("Processed file [%d/%d]", image_index,
                                num_images_)
                << std::endl;

      std::cout << StringPrintf("  Name:            %s",
                                image_data.image.Name().c_str())
                << std::endl;

      if (image_data.status == ImageReader::Status::IMAGE_EXISTS) {
        std::cout << "  SKIP: Features for image already extracted."
                  << std::endl;
      } else if (image_data.status == ImageReader::Status::BITMAP_ERROR) {
        std::cout << "  ERROR: Failed to read image file format." << std::endl;
      } else if (image_data.status ==
                 ImageReader::Status::CAMERA_SINGLE_DIM_ERROR) {
        std::cout << "  ERROR: Single camera specified, "
                     "but images have different dimensions."
                  << std::endl;
      } else if (image_data.status ==
                 ImageReader::Status::CAMERA_EXIST_DIM_ERROR) {
        std::cout << "  ERROR: Image previously processed, but current image "
                     "has different dimensions."
                  << std::endl;
      } else if (image_data.status == ImageReader::Status::CAMERA_PARAM_ERROR) {
        std::cout << "  ERROR: Camera has invalid parameters." << std::endl;
      } else if (image_data.status == ImageReader::Status::FAILURE) {
        std::cout << "  ERROR: Failed to extract features." << std::endl;
      }

      if (image_data.status != ImageReader::Status::SUCCESS) {
        continue;
      }

      std::cout << StringPrintf("  Dimensions:      %d x %d",
                                image_data.camera.Width(),
                                image_data.camera.Height())
                << std::endl;
      std::cout << StringPrintf("  Camera:          #%d - %s",
                                image_data.camera.CameraId(),
                                image_data.camera.ModelName().c_str())
                << std::endl;
      std::cout << StringPrintf("  Focal Length:    %.2fpx",
                                image_data.camera.MeanFocalLength());
      if (image_data.camera.HasPriorFocalLength()) {
        std::cout << " (Prior)" << std::endl;
      } else {
        std::cout << std::endl;
      }
      if (image_data.image.HasTvecPrior()) {
        std::cout << StringPrintf(
                         "  GPS:             LAT=%.3f, LON=%.3f, ALT=%.3f",
                         image_data.image.TvecPrior(0),
                         image_data.image.TvecPrior(1),
                         image_data.image.TvecPrior(2))
                  << std::endl;
      }
      std::cout << StringPrintf("  Features:        %d",
                                image_data.keypoints.size())
                << std::endl;

      DatabaseTransaction database_transaction(database_);

      if (image_data.image.ImageId() == kInvalidImageId) {
        image_data.image.SetImageId(database_->WriteImage(image_data.image));
      }

      if (!database_->ExistsKeypoints(image_data.image.ImageId())) {
        database_->WriteKeypoints(image_data.image.ImageId(),
                                  image_data.keypoints);
      }

      if (!database_->ExistsDescriptors(image_data.image.ImageId())) {
        database_->WriteDescriptors(image_data.image.ImageId(),
                                    image_data.descriptors);
      }
    } else {
      break;
    }
  }
}

}  // namespace internal
}  // namespace colmap
