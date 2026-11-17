#include "detect_net.h"

DetectNet::DetectNet(std::string, std::string, int, float)
{
}

DetectNet::~DetectNet()
{
}

std::vector<image_recognition_interfaces::msg::Recognition> DetectNet::processImage(const cv::Mat &)
{
  return {};
}
