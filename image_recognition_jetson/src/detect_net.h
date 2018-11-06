#include <opencv/cv.h>
#include <vector>

#include <image_recognition_msgs/Recognitions.h>

class DetectNet
{

public:
  DetectNet(std::string prototxt_path, std::string model_path, int  mean_pixel, float threshold);

  ~DetectNet();

  std::vector<image_recognition_msgs::Recognition> processImage(const cv::Mat& cv_im);

private:
  int mean_pixel_;
  float threshold_;

};
