#ifndef COLMAP_SRC_UTIL_PROGRESS_BAR_H_
#define COLMAP_SRC_UTIL_PROGRESS_BAR_H_

#include <functional>
#include <iostream>
#include <string>
#include <mutex>

namespace colmap {

/**
 * 进度条回调函数类型定义
 * @param current 当前进度值
 * @param total 总进度值  
 * @param message 进度消息描述
 */
using ProgressCallback = std::function<void(size_t current, size_t total, const std::string& message)>;

/**
 * 控制台进度条显示类
 * 用于在控制台显示实时更新的进度条
 */
class ConsoleProgressBar {
 public:
  /**
   * 构造函数
   * @param width 进度条宽度（字符数）
   * @param show_percentage 是否显示百分比
   * @param show_bar 是否显示进度条
   */
  ConsoleProgressBar(int width = 50, bool show_percentage = true, bool show_bar = true);

  /**
   * 更新进度
   * @param current 当前进度值
   * @param total 总进度值
   * @param message 进度消息
   */
  void Update(size_t current, size_t total, const std::string& message = "");

  /**
   * 完成进度，显示完成状态
   * @param message 完成消息
   */
  void Finish(const std::string& message = "");

  /**
   * 重置进度条状态
   */
  void Reset();

 private:
  int width_;                  // 进度条宽度
  bool show_percentage_;       // 是否显示百分比
  bool show_bar_;             // 是否显示进度条
  size_t last_current_;       // 上次的当前值，避免频繁重绘
  std::mutex mutex_;          // 线程安全保护
  bool finished_;             // 是否已完成

  /**
   * 生成进度条字符串
   * @param current 当前进度
   * @param total 总进度
   * @return 进度条字符串
   */
  std::string GenerateBar(size_t current, size_t total) const;
};

/**
 * 多阶段进度管理器
 * 管理多个连续执行的阶段，每个阶段有独立的进度
 */
class MultiStageProgressManager {
 public:
  /**
   * 构造函数
   * @param stage_names 各阶段名称列表
   * @param stage_weights 各阶段权重（可选，用于计算总体进度）
   */
  MultiStageProgressManager(const std::vector<std::string>& stage_names,
                           const std::vector<double>& stage_weights = {});

  /**
   * 开始指定阶段
   * @param stage_index 阶段索引
   * @param total_items 该阶段总项目数
   */
  void StartStage(size_t stage_index, size_t total_items);

  /**
   * 更新当前阶段进度
   * @param current 当前完成数量
   * @param message 进度消息
   */
  void UpdateCurrentStage(size_t current, const std::string& message = "");

  /**
   * 完成当前阶段
   */
  void FinishCurrentStage();

  /**
   * 获取当前阶段的进度回调函数
   * @return 进度回调函数
   */
  ProgressCallback GetCurrentStageCallback();

 private:
  std::vector<std::string> stage_names_;    // 阶段名称
  std::vector<double> stage_weights_;       // 阶段权重
  size_t current_stage_;                    // 当前阶段索引
  size_t current_stage_total_;              // 当前阶段总项目数
  ConsoleProgressBar progress_bar_;         // 进度条显示器
  std::mutex mutex_;                        // 线程安全保护
};

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_PROGRESS_BAR_H_
