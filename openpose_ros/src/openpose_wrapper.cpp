#include "openpose_wrapper.h"

//#define USE_CAFFE

//#include <std_srvs/Empty.h>
//#include <ros/node_handle.h>
//#include <ros/service_server.h>
//#include <ros/init.h>

////#include <openpose/pose/poseExtractor.hpp>
////#include <openpose/pose/poseExtractorCaffe.hpp>
////#include <openpose/pose/poseParameters.hpp>

////#include <openpose/core/headers.hpp>
////#include <openpose/filestream/headers.hpp>
////#include <openpose/gui/headers.hpp>
////#include <openpose/pose/headers.hpp>
////#include <openpose/utilities/headers.hpp>

namespace openpose_ros
{

//op::PoseModel stringToPoseModel(const std::string& pose_model_string)
//{
//  if (pose_model_string == "COCO")
//    return op::PoseModel::COCO_18;
//  else if (pose_model_string == "MPI")
//    return op::PoseModel::MPI_15;
//  else if (pose_model_string == "MPI_4_layers")
//    return op::PoseModel::MPI_15_4;
//  else
//  {
//    ROS_ERROR("String does not correspond to any model (COCO, MPI, MPI_4_layers)");
//    return op::PoseModel::COCO_18;
//  }
//}

//std::map<unsigned char, std::string> getBodyPartMapFromPoseModel(const op::PoseModel& pose_model)
//{
//  if (pose_model == op::PoseModel::COCO_18)
//  {
//    return op::POSE_COCO_BODY_PARTS;
//  }
//  else if (pose_model == op::PoseModel::MPI_15 || pose_model == op::PoseModel::MPI_15_4)
//  {
//    return op::POSE_MPI_BODY_PARTS;
//  }
//  else
//  {
//    ROS_FATAL("Invalid pose model, not map present");
//    exit(1);
//  }
//}

//OpenposeWrapper::OpenposeWrapper(const cv::Size& net_input_size, const cv::Size &net_output_size,
//                                 const cv::Size& output_size, size_t num_scales, double scale_gap,
//                                 size_t num_gpu_start, const std::string& model_folder, const std::string& pose_model) :
//  net_input_size_(net_input_size),
//  net_output_size_(net_output_size),
//  num_scales_(num_scales),
//  scale_gap_(scale_gap),
//  num_gpu_start_(num_gpu_start),
//  bodypart_map_(getBodyPartMapFromPoseModel(stringToPoseModel(pose_model_))),
//  pose_extractor_(std::shared_ptr<op::PoseExtractorCaffe>(new op::PoseExtractorCaffe(net_input_size_, net_output_size_, output_size, num_scales_,
//                                                                                     scale_gap_, stringToPoseModel(pose_model_), model_folder, num_gpu_start_)))
//{
//}

//bool OpenposeWrapper::detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions)
//{
//  ROS_INFO_STREAM("Perform forward pass with the following settings:");
//  ROS_INFO_STREAM("- net_input_size: " << net_input_size_);
//  ROS_INFO_STREAM("- num_scales: " << num_scales_);
//  ROS_INFO_STREAM("- scale_gap: " << scale_gap_);
//  ROS_INFO_STREAM("- image_size: " << image.size());
//  op::CvMatToOpInput cv_mat_to_op_input(net_input_size_, num_scales_, scale_gap_);

//  pose_extractor_->forwardPass(cv_mat_to_op_input.format(image), image.size());
//  ROS_INFO("pose_extractor->forwardPass done");

//  const auto pose_keypoints = g_pose_extractor->getPoseKeyPoints();
//  pose_renderer.renderPose(output_array, pose_keypoints);

//  if (!pose_keypoints.empty() && pose_keypoints.getNumberDimensions() != 3)
//  {
//    ROS_ERROR("pose.getNumberDimensions(): %d != 3", (size_t) pose_keypoints.getNumberDimensions());
//    return false;
//  }

//  size_t num_people = pose_keypoints.getSize(0);
//  size_t num_bodyparts = pose_keypoints.getSize(1);
//  recognitions.resize(num_people * num_bodyparts);

//  ROS_INFO("Detected %d persons", (size_t) num_people);

//  for (size_t person_idx = 0; person_idx < num_people; person_idx++)
//  {
//    for (size_t bodypart_idx = 0; bodypart_idx < num_bodyparts; bodypart_idx++)
//    {
//      size_t index = (person_idx * num_bodyparts + bodypart_idx);

//      recognitions[index].group_id = person_idx;
//      recognitions[index].roi.width = 1;
//      recognitions[index].roi.heigth = 1;
//      recognitions[index].categorical_distribution.resize(1);
//      recognitions[index].categorical_distribution.back().label = bodypart_map[bodypart_idx];

//      recognitions[index].roi.x_offset = pose_keypoints[3 * index];
//      recognitions[index].roi.y_offset = pose_keypoints[3 * index + 1];
//      recognitions[index].categorical_distribution.back().probability = pose_keypoints[3 * index + 2];
//    }
//  }

//  return true;
//}

OpenposeWrapper::OpenposeWrapper(const cv::Size& net_input_size, const cv::Size &net_output_size,
                                 const cv::Size& output_size, size_t num_scales, double scale_gap,
                                 size_t num_gpu_start, const std::string& model_folder, const std::string& pose_model)
{
}

bool OpenposeWrapper::detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions)
{
  return true;
}

}

