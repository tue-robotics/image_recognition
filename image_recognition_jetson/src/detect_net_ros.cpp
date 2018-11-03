#include "detect_net.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <image_recognition_msgs/Recognize.h>

//!
//! \brief g_detect_net Pointer to detect network
//!
std::shared_ptr<DetectNet> g_detect_net;

//!
//! \brief g_recognitions_publisher Recognition result publisher when triggered via topic
//!
ros::Publisher g_recognitions_publisher;

//!
//! \brief processImage Process incoming image via topic or srv call
//! \param input Input image opencv
//! \return The recognitions
//!
std::vector<image_recognition_msgs::Recognition> processImage(const sensor_msgs::Image& input)
{
  ros::Time start = ros::Time::now();

  cv::Mat cv_im = cv_bridge::toCvCopy(input, "bgr8")->image;

  // convert bit depth
  cv_im.convertTo(cv_im, CV_32FC3);

  // convert color
  cv::cvtColor(cv_im, cv_im, CV_BGR2RGBA);

  std::vector<image_recognition_msgs::Recognition> recognitions = g_detect_net->processImage(cv_im);

  ROS_INFO_STREAM("processImage took " << (ros::Time::now() - start).toSec() << " seconds");

  return recognitions;
}

//!
//! \brief callback Topic callback
//!
void callback(const sensor_msgs::ImageConstPtr& input)
{
  image_recognition_msgs::Recognitions msg;
  msg.header = input->header;
  msg.recognitions = processImage(*input);
  g_recognitions_publisher.publish(msg);
}

//!
//! \brief srvCallback Service call callback
//!
bool srvCallback(image_recognition_msgs::Recognize::Request& req, image_recognition_msgs::Recognize::Response& res)
{
  res.recognitions = processImage(req.image);
  return true;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "detect_net");

  // get node handles
  ros::NodeHandle local_nh("~");
  ros::NodeHandle nh;

  // Load required parameters
  std::string prototxt_path;
  std::string model_path;
  if (!local_nh.getParam("prototxt_path", prototxt_path))
  {
    ROS_FATAL("Please specify ~prototxt_path");
    return 1;
  }
  if (!local_nh.getParam("model_path", model_path))
  {
    ROS_FATAL("Please specify ~model_path");
    return 1;
  }

  try
  {
    g_detect_net = std::shared_ptr<DetectNet>(new DetectNet(prototxt_path, model_path,
                                                            local_nh.param("mean_pixel", 0.0),
                                                            local_nh.param("threshold", 0.5)));

    // setup image transport
    image_transport::ImageTransport it(nh);
  
    // subscriber for passing in images
    image_transport::Subscriber image_subscriber = it.subscribe("image", 10, &callback);
    ros::ServiceServer recognition_service = nh.advertiseService("recognize", &srvCallback);
  
    g_recognitions_publisher = nh.advertise<image_recognition_msgs::Recognitions>("recognitions", 1);

    ROS_INFO("DetectNetROS initialized, spinning ...");

    ros::spin();
  }
  catch (const std::exception& e)
  {
    ROS_FATAL_STREAM("Error: " << e.what());
    return 1;
  }

  return 0;
}
