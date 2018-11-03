#include "detect_net.h"

DetectNet::DetectNet(std::string prototxt_path, std::string model_path, int mean_pixel, float threshold)
{
}

DetectNet::~DetectNet()
{
}

std::vector<image_recognition_msgs::Recognition> DetectNet::processImage(const cv::Mat &cv_im)
{
  return std::vector<image_recognition_msgs::Recognition>();
}
