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

OpenposeWrapper::OpenposeWrapper(const cv::Size& net_input_size, const cv::Size &net_output_size,
                                 const cv::Size& output_size, size_t num_scales, double scale_gap,
                                 size_t num_gpu_start, const std::string& model_folder,
                                 const std::string& pose_model, double overlay_alpha) :
  net_input_size_(net_input_size),
  net_output_size_(net_output_size),
  num_scales_(num_scales),
  scale_gap_(scale_gap),
  bodypart_map_(getBodyPartMapFromPoseModel(stringToPoseModel(pose_model))),
  pose_extractor_(std::shared_ptr<op::PoseExtractorCaffe>(
                    new op::PoseExtractorCaffe(op::Point<int>(net_input_size_.width, net_input_size_.height),
                                               op::Point<int>(net_output_size_.width, net_output_size_.height),
                                               op::Point<int>(output_size.width, output_size.height),
                                               num_scales_, stringToPoseModel(pose_model), model_folder, num_gpu_start))),
  pose_renderer_(std::shared_ptr<op::PoseRenderer>(
                   new op::PoseRenderer(op::Point<int>(net_output_size_.width, net_output_size_.height),
                                        op::Point<int>(output_size.width, output_size.height),
                                        stringToPoseModel(pose_model),
                                        nullptr,
                                        (float) overlay_alpha)))
{
  pose_renderer_->initializationOnThread();
}

bool OpenposeWrapper::detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions, cv::Mat& overlayed_image)
{
  // Step 3 - Initialize all required classes
  op::CvMatToOpInput cv_mat_to_input(op::Point<int>(net_input_size_.width, net_input_size_.height), (int) num_scales_, scale_gap_);
  op::CvMatToOpOutput cv_mat_to_output(op::Point<int>(output_size_.width, output_size_.height));
  op::OpOutputToCvMat op_output_to_cv_mat(op::Point<int>(output_size_.width, output_size_.height));

  // Step 2 - Format input image to OpenPose input and output formats
  //const auto net_input_array = cv_mat_to_input.format(image);
  double scale_input_to_output;
  op::Array<float> net_input_array;
  std::vector<float> scale_ratios;
  std::tie(net_input_array, scale_ratios) = cv_mat_to_input.format(image);
  op::Array<float> output_array;
  std::tie(scale_input_to_output, output_array) = cv_mat_to_output.format(image);
  // Step 3 - Estimate poseKeyPoints
  pose_extractor_->forwardPass(net_input_array, {image.cols, image.rows}, scale_ratios);
  const auto pose_keypoints = pose_extractor_->getPoseKeypoints();

  // Step 4 - Render poseKeyPoints
  pose_renderer_->renderPose(output_array, pose_keypoints);

  // Step 5 - OpenPose output format to cv::Mat
  overlayed_image = op_output_to_cv_mat.formatToCvMat(output_array);

  size_t num_people = pose_keypoints.getSize(0);
  size_t num_bodyparts = pose_keypoints.getSize(1);
  recognitions.resize(num_people * num_bodyparts);

  ROS_INFO("Detected %d persons", (int) num_people);

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

      recognitions[index].roi.x_offset = pose_keypoints[3 * index];
      recognitions[index].roi.y_offset = pose_keypoints[3 * index + 1];
      recognitions[index].categorical_distribution.probabilities.front().probability = pose_keypoints[3 * index + 2];
    }
  }

  return true;
}

