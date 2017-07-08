#include "openpose_wrapper.h"

#include <image_recognition_msgs/Recognize.h>
#include <ros/node_handle.h>

std::shared_ptr<openpose_ros::OpenposeWrapper> g_openpose_wrapper_;

//!
//! \brief getParam Get parameter from node handle
//! \param nh The nodehandle
//! \param param_name Key string
//! \param default_value Default value if not found
//! \return The parameter value
//!
template <typename T>
T getParam(const ros::NodeHandle& nh, const std::string& param_name, T default_value)
{
  T value;
  if (nh.hasParam(param_name))
  {
    nh.getParam(param_name, value);
  }
  else
  {
    ROS_WARN_STREAM("Parameter '" << param_name << "' not found, defaults to '" << default_value << "'");
    value = default_value;
  }
  return value;
}

bool detectPosesCallback(image_recognition_msgs::Recognize::Request& req, image_recognition_msgs::Recognize::Response& res)
{
  ROS_INFO("detectPosesCallback");

  // Convert ROS message to opencv image
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(req.image, req.image.encoding);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("detectPosesCallback cv_bridge exception: %s", e.what());
    return false;
  }
  cv::Mat image = cv_ptr->image;
  if(image.empty())
  {
    ROS_ERROR("Empty image!");
    return false;
  }

  return g_openpose_wrapper_->detectPoses(image, res.recognitions);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "openpose");

  ros::NodeHandle local_nh("~");
  g_openpose_wrapper_ = std::shared_ptr<openpose_ros::OpenposeWrapper>(
        new openpose_ros::OpenposeWrapper(cv::Size(getParam(local_nh, "net_input_width", 656), getParam(local_nh, "net_input_height", 368)),
                                          cv::Size(getParam(local_nh, "net_output_width", 656), getParam(local_nh, "net_output_height", 368)),
                                          cv::Size(getParam(local_nh, "output_width", 1280), getParam(local_nh, "output_height", 720)),
                                          getParam(local_nh, "num_scales", 1),
                                          getParam(local_nh, "scale_gap", 0.3),
                                          getParam(local_nh, "num_gpu_start", 0),
                                          getParam(local_nh, "model_folder", std::string("~/openpose/models/")),
                                          getParam(local_nh, "pose_model", std::string("COCO"))));

  ros::NodeHandle nh;
  ros::ServiceServer service = nh.advertiseService("recognize", detectPosesCallback);

  ros::spin();

  return 0;
}
