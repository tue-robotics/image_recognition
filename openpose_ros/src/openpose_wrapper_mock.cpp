#include "openpose_wrapper.h"
#include <ros/console.h>

//!
//! \brief getRecognition Returns dummy recognition for simulation purposes
//! \param x x_offset
//! \param y y_offset
//! \param label Label
//! \return Returns a recognition
//!
image_recognition_msgs::Recognition getRecognition(size_t x, size_t y, const std::string& label)
{
  image_recognition_msgs::Recognition r;
  r.roi.x_offset = x;
  r.roi.y_offset = y;
  r.roi.width = 1;
  r.roi.height = 1;
  r.categorical_distribution.probabilities.resize(1);
  r.categorical_distribution.probabilities.back().label = label;
  r.categorical_distribution.probabilities.back().probability = 1.0;
  return r;
}

OpenposeWrapper::OpenposeWrapper(const cv::Size& net_input_size, const cv::Size& net_output_size,
                                 size_t num_scales, double scale_gap,
                                 size_t num_gpu_start, const std::string& model_folder,
                                 const std::string& pose_model, double overlay_alpha)
{
  ROS_WARN("OpenposeWrapper::OpenposeWrapper -- Using Mock!");
}

bool OpenposeWrapper::detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions, cv::Mat& overlayed_image)
{
  size_t x = image.cols / 2;
  size_t y = image.rows / 2;

  overlayed_image = image;

  return true;
}
