#include <opencv2/opencv.hpp>
#include <vector>

#include <image_recognition_interfaces/msg/recognition.hpp>

class DetectNet
{

public:
  DetectNet(std::string prototxt_path, std::string model_path, int  mean_pixel, float threshold);

  ~DetectNet();

  std::vector<image_recognition_interfaces::msg::Recognition> processImage(const cv::Mat& cv_im);

private:
  int mean_pixel_;
  float threshold_;

};
