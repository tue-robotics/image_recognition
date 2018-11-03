#ifndef OPENPOSE_WRAPPER_H
#define OPENPOSE_WRAPPER_H

#include <cv_bridge/cv_bridge.h>
#include <image_recognition_msgs/Recognition.h>

//! Forward declare openpose
namespace op
{
  class PoseExtractor;
  class PoseRenderer;
}

class OpenposeWrapper
{
public:
  //!
  //! \brief OpenposeWrapper Wraps the openpose implementation (this way we can use a dummy in simulation)
  //! \param net_input_size Input size of the network
  //! \param net_output_size Network output size
  //! \param num_scales Number of scales to average
  //! \param scale_gap Scale gap between scales
  //! \param num_gpu_start GPU device start number
  //! \param model_folder Where to find the openpose models
  //! \param pose_model Pose model string
  //! \param overlay_alpha Alpha factor used for overlaying the image
  //!
  OpenposeWrapper(const cv::Size& net_input_size, const cv::Size& net_output_size, size_t num_scales, 
                  double scale_gap, size_t num_gpu_start, const std::string &model_folder,
                  const std::string& pose_model, double overlay_alpha);

  bool detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions, cv::Mat& overlayed_image);

private:
  std::shared_ptr<op::PoseRenderer> pose_renderer_;
  std::shared_ptr<op::PoseExtractor> pose_extractor_;

  std::map<unsigned int, std::string> bodypart_map_;

  cv::Size net_input_size_;
  cv::Size net_output_size_;

  size_t num_scales_;
  double scale_gap_;

};

#endif // OPENPOSE_WRAPPER_H
