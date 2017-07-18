#include "openpose_wrapper.h"

#include <ros/console.h>

#include <openpose/pose/poseExtractor.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>
#include <openpose/pose/poseParameters.hpp>

#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

//!
//! \brief stringToPoseModel Returns a pose model based on a string
//! \param pose_model_string Pose model string
//! \return Pose model
//!
op::PoseModel stringToPoseModel(const std::string& pose_model_string)
{
  if (pose_model_string == "COCO")
    return op::PoseModel::COCO_18;
  else if (pose_model_string == "MPI")
    return op::PoseModel::MPI_15;
  else if (pose_model_string == "MPI_4_layers")
    return op::PoseModel::MPI_15_4;
  else
  {
    ROS_ERROR("String does not correspond to any model (COCO, MPI, MPI_4_layers)");
    return op::PoseModel::COCO_18;
  }
}

//!
//! \brief getBodyPartMapFromPoseModel Returns the body part map from pose model (string mapping)
//! \param pose_model Pose model input
//! \return String map
//!
std::map<unsigned int, std::string> getBodyPartMapFromPoseModel(const op::PoseModel& pose_model)
{
  if (pose_model == op::PoseModel::COCO_18)
  {
    return op::POSE_COCO_BODY_PARTS;
  }
  else if (pose_model == op::PoseModel::MPI_15 || pose_model == op::PoseModel::MPI_15_4)
  {
    return op::POSE_MPI_BODY_PARTS;
  }
  else
  {
    ROS_FATAL("Invalid pose model, not map present");
    exit(1);
  }
}

OpenposeWrapper::OpenposeWrapper(const cv::Size& net_input_size, const cv::Size& net_output_size,
                                 size_t num_scales, double scale_gap,
                                 size_t num_gpu_start, const std::string& model_folder,
                                 const std::string& pose_model, double overlay_alpha) :
  net_input_size_(net_input_size),
  net_output_size_(net_output_size),
  num_scales_(num_scales),
  scale_gap_(scale_gap),
  bodypart_map_(getBodyPartMapFromPoseModel(stringToPoseModel(pose_model)))
{
  op::ConfigureLog::setPriorityThreshold(op::Priority::High); 

  op::Point<int> op_net_input_size(net_input_size_.width, net_input_size_.height);
  op::Point<int> op_net_output_size(net_output_size_.width, net_output_size_.height);

  ROS_INFO("Net input size: [%d x %d]", op_net_input_size.x, op_net_input_size.y);
  ROS_INFO("Net output size: [%d x %d]", op_net_output_size.x, op_net_output_size.y);

  pose_extractor_ = std::shared_ptr<op::PoseExtractorCaffe>(
                    new op::PoseExtractorCaffe(op_net_input_size, op_net_output_size, op_net_output_size,
                                               num_scales_, stringToPoseModel(pose_model), model_folder, num_gpu_start));

  pose_renderer_ = std::shared_ptr<op::PoseRenderer>(
                   new op::PoseRenderer(op_net_output_size, op_net_output_size,
                                        stringToPoseModel(pose_model),
                                        nullptr,
                                        (float) overlay_alpha));

  pose_extractor_->initializationOnThread();
  pose_renderer_->initializationOnThread();
}

bool OpenposeWrapper::detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions, cv::Mat& overlayed_image)
{
  ROS_INFO("OpenposeWrapper::detectPoses: Detecting poses on image of size [%d x %d]", image.cols, image.rows);

  op::Point<int> op_net_input_size(net_input_size_.width, net_input_size_.height);
  op::CvMatToOpInput cv_mat_to_input(op_net_input_size, (int) num_scales_, scale_gap_);
  op::Array<float> net_input_array;
  std::vector<float> scale_ratios;
  std::tie(net_input_array, scale_ratios) = cv_mat_to_input.format(image);

  ROS_INFO("OpenposeWrapper::detectPoses: Net input size: [%d x %d]", op_net_input_size.x, op_net_input_size.y);

  op::Point<int> op_net_output_size(net_output_size_.width, net_output_size_.height);
  op::OpOutputToCvMat op_output_to_cv_mat(op_net_output_size);
  op::CvMatToOpOutput cv_mat_to_output(op_net_output_size);

  ROS_INFO("OpenposeWrapper::detectPoses: Net output size: [%d x %d]", op_net_output_size.x, op_net_output_size.y);

  op::Array<float> output_array;
  double scale_input_to_output;
  std::tie(scale_input_to_output, output_array) = cv_mat_to_output.format(image);

  ROS_INFO("OpenposeWrapper::detectPoses: Applying forward pass on image of size: [%d x %d]", image.cols, image.rows);

  // Step 3 - Estimate poseKeyPoints
  pose_extractor_->forwardPass(net_input_array, {image.cols, image.rows}, scale_ratios);
  const auto pose_keypoints = pose_extractor_->getPoseKeypoints();

  size_t num_people = pose_keypoints.getSize(0);
  size_t num_bodyparts = pose_keypoints.getSize(1);

  ROS_INFO("OpenposeWrapper::detectPoses: Rendering %d keypoints", (int) (num_people * num_bodyparts));

  // Step 4 - Render poseKeyPoints
  pose_renderer_->renderPose(output_array, pose_keypoints);

  // Step 5 - OpenPose output format to cv::Mat
  overlayed_image = op_output_to_cv_mat.formatToCvMat(output_array);

  // Calculate the factors between the input image and the output image
  double width_factor = (double) image.cols / overlayed_image.cols;
  double height_factor = (double) image.rows / overlayed_image.rows;
  double scale_factor = std::fmax(width_factor, height_factor);

  recognitions.resize(num_people * num_bodyparts);

  ROS_INFO("OpenposeWrapper::detectPoses: Detected %d persons", (int) num_people);

  for (size_t person_idx = 0; person_idx < num_people; person_idx++)
  {
    for (size_t bodypart_idx = 0; bodypart_idx < num_bodyparts; bodypart_idx++)
    {
      size_t index = (person_idx * num_bodyparts + bodypart_idx);

      recognitions[index].group_id = person_idx;
      recognitions[index].roi.width = 1;
      recognitions[index].roi.height = 1;
      recognitions[index].categorical_distribution.probabilities.resize(1);
      recognitions[index].categorical_distribution.probabilities.front().label = bodypart_map_[bodypart_idx];

      recognitions[index].roi.x_offset = pose_keypoints[3 * index] * scale_factor;
      recognitions[index].roi.y_offset = pose_keypoints[3 * index + 1] * scale_factor;
      recognitions[index].categorical_distribution.probabilities.front().probability = pose_keypoints[3 * index + 2];
    }
  }

  return true;
}

