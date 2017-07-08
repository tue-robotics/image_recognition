#ifndef OPENPOSE_WRAPPER_H
#define OPENPOSE_WRAPPER_H

#include <cv_bridge/cv_bridge.h>
#include <image_recognition_msgs/Recognition.h>

namespace op
{
  class PoseExtractor;
  class PoseRenderer;
}

class OpenposeWrapper
{
public:
  OpenposeWrapper(const cv::Size& net_input_size, const cv::Size& net_output_size, const cv::Size &output_size,
                  size_t num_scales, double scale_gap, size_t num_gpu_start, const std::string &model_folder,
                  const std::string& pose_model, double overlay_alpha);

  bool detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions, cv::Mat& overlayed_image);

private:
  std::shared_ptr<op::PoseRenderer> pose_renderer_;
  std::shared_ptr<op::PoseExtractor> pose_extractor_;

  std::map<unsigned int, std::string> bodypart_map_;

  cv::Size net_input_size_;
  cv::Size net_output_size_;
  cv::Size output_size_;
  size_t num_scales_;
  double scale_gap_;
  double alpha_pose_;

};

#endif // OPENPOSE_WRAPPER_H
