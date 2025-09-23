#include "util/progress_bar.h"

#include <iomanip>
#include <sstream>
#include <algorithm>

namespace colmap {

ConsoleProgressBar::ConsoleProgressBar(int width, bool show_percentage, bool show_bar)
    : width_(width),
      show_percentage_(show_percentage),
      show_bar_(show_bar),
      last_current_(SIZE_MAX),
      finished_(false) {}

void ConsoleProgressBar::Update(size_t current, size_t total, const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  // 避免频繁重绘，只有在进度真正变化时才更新
  if (current == last_current_ && message.empty()) {
    return;
  }
  
  last_current_ = current;
  
  // 清除当前行并回到行首
  std::cout << "\r";
  
  std::ostringstream oss;
  
  // 添加消息前缀
  if (!message.empty()) {
    oss << message << " ";
  }
  
  // 显示进度条
  if (show_bar_ && total > 0) {
    oss << GenerateBar(current, total) << " ";
  }
  
  // 显示百分比
  if (show_percentage_ && total > 0) {
    double percentage = (double)current / total * 100.0;
    oss << std::fixed << std::setprecision(1) << percentage << "% ";
  }
  
  // 显示数值进度
  oss << "[" << current << "/" << total << "]";
  
  std::cout << oss.str() << std::flush;
}

void ConsoleProgressBar::Finish(const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (finished_) {
    return;
  }
  
  finished_ = true;
  
  // 清除当前行并显示完成信息
  std::cout << "\r";
  if (!message.empty()) {
    std::cout << message;
  } else {
    std::cout << "完成";
  }
  std::cout << std::endl;
}

void ConsoleProgressBar::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  last_current_ = SIZE_MAX;
  finished_ = false;
}

std::string ConsoleProgressBar::GenerateBar(size_t current, size_t total) const {
  if (total == 0) {
    return std::string(width_, '-');
  }
  
  double progress = (double)current / total;
  int filled_width = (int)(progress * width_);
  
  std::string bar = "[";
  for (int i = 0; i < width_; ++i) {
    if (i < filled_width) {
      bar += "█";  // 已完成部分
    } else {
      bar += "░";  // 未完成部分
    }
  }
  bar += "]";
  
  return bar;
}

/**
 * [功能描述]：多阶段进度管理器的构造函数，用于管理包含多个阶段的复杂任务进度。
 * @param stage_names：阶段名称向量，包含每个阶段的描述性名称。
 * @param stage_weights：阶段权重向量，用于计算各阶段在总进度中的占比，可为空。
 * @return 无返回值：构造函数。
 */
MultiStageProgressManager::MultiStageProgressManager(
    const std::vector<std::string>& stage_names,        // 阶段名称列表
    const std::vector<double>& stage_weights)            // 各阶段权重列表
    : stage_names_(stage_names),                         // 初始化阶段名称向量
      stage_weights_(stage_weights),                     // 初始化阶段权重向量
      current_stage_(0),                                 // 当前阶段索引，初始为0
      current_stage_total_(0),                          // 当前阶段总任务数，初始为0
      progress_bar_(40, true, true) {                   // 创建进度条对象，宽度40，显示百分比和进度条
  
  // 如果没有提供权重，则默认每个阶段权重相等
  if (stage_weights_.empty()) {
    stage_weights_.assign(stage_names_.size(), 1.0);    // 为每个阶段分配权重1.0
  }
  
  // 确保权重向量大小与阶段数量一致
  if (stage_weights_.size() != stage_names_.size()) {
    stage_weights_.resize(stage_names_.size(), 1.0);    // 调整权重向量大小，不足部分用1.0填充
  }
}

void MultiStageProgressManager::StartStage(size_t stage_index, size_t total_items) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (stage_index >= stage_names_.size()) {
    return;
  }
  
  // 如果前一阶段未完成，先完成它
  if (current_stage_ < stage_names_.size()) {
    progress_bar_.Finish();
  }
  
  current_stage_ = stage_index;
  current_stage_total_ = total_items;
  
  // 重置进度条并显示新阶段开始信息
  progress_bar_.Reset();
  
  std::string stage_message = "阶段 " + std::to_string(stage_index + 1) + "/" + 
                             std::to_string(stage_names_.size()) + ": " + stage_names_[stage_index];
  
  std::cout << "\n" << stage_message << std::endl;
}

void MultiStageProgressManager::UpdateCurrentStage(size_t current, const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (current_stage_ >= stage_names_.size()) {
    return;
  }
  
  std::string full_message = message.empty() ? stage_names_[current_stage_] : message;
  progress_bar_.Update(current, current_stage_total_, full_message);
}

void MultiStageProgressManager::FinishCurrentStage() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (current_stage_ >= stage_names_.size()) {
    return;
  }
  
  std::string finish_message = stage_names_[current_stage_] + " 完成";
  progress_bar_.Finish(finish_message);
}

ProgressCallback MultiStageProgressManager::GetCurrentStageCallback() {
  return [this](size_t current, size_t total, const std::string& message) {
    // 如果总数与当前阶段设定的不同，更新总数
    if (total != current_stage_total_ && total > 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      current_stage_total_ = total;
    }
    UpdateCurrentStage(current, message);
  };
}

}  // namespace colmap
