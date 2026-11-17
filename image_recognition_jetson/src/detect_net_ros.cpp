#include "detect_net.h"

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <image_recognition_interfaces/msg/recognitions.hpp>
#include <image_recognition_interfaces/srv/recognize.hpp>

namespace
{
using image_recognition_interfaces::msg::Recognition;
using image_recognition_interfaces::msg::Recognitions;
using image_recognition_interfaces::srv::Recognize;
using sensor_msgs::msg::Image;

class DetectNetNode : public rclcpp::Node
{
public:
  DetectNetNode() : Node("detect_net")
  {
    const auto prototxt_path = declare_parameter<std::string>("prototxt_path", "");
    const auto model_path = declare_parameter<std::string>("model_path", "");
    const auto mean_pixel = declare_parameter<double>("mean_pixel", 0.0);
    const auto threshold = declare_parameter<double>("threshold", 0.5);
    if (prototxt_path.empty() || model_path.empty())
    {
      throw std::invalid_argument("Set both the prototxt_path and model_path parameters");
    }

    detector_ = std::make_unique<DetectNet>(prototxt_path, model_path, mean_pixel, threshold);
    publisher_ = create_publisher<Recognitions>("recognitions", 1);
    const auto transport = declare_parameter<std::string>("image_transport", "raw");
    subscription_ = image_transport::create_subscription(
      this, "image", [this](const Image::ConstSharedPtr& input) { publish_recognitions(*input); },
      transport, rclcpp::SensorDataQoS().keep_last(10).get_rmw_qos_profile());
    service_ = create_service<Recognize>(
      "recognize", [this](const std::shared_ptr<Recognize::Request> request,
        std::shared_ptr<Recognize::Response> response) {
        response->recognitions = process_image(request->image);
      });
  }

private:
  std::vector<Recognition> process_image(const Image& input)
  {
    const auto start = now();
    try
    {
      auto image = cv_bridge::toCvCopy(input, "bgr8")->image;
      image.convertTo(image, CV_32FC3);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
      auto recognitions = detector_->processImage(image);
      RCLCPP_INFO(get_logger(), "processImage took %.3f seconds", (now() - start).seconds());
      return recognitions;
    }
    catch (const std::exception& error)
    {
      RCLCPP_ERROR(get_logger(), "Unable to process image: %s", error.what());
      return {};
    }
  }

  void publish_recognitions(const Image& image)
  {
    Recognitions result;
    result.header = image.header;
    result.recognitions = process_image(image);
    publisher_->publish(result);
  }

  std::unique_ptr<DetectNet> detector_;
  rclcpp::Publisher<Recognitions>::SharedPtr publisher_;
  image_transport::Subscriber subscription_;
  rclcpp::Service<Recognize>::SharedPtr service_;
};
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  int result = 0;
  try
  {
    rclcpp::spin(std::make_shared<DetectNetNode>());
  }
  catch (const std::exception& error)
  {
    RCLCPP_FATAL(rclcpp::get_logger("detect_net"), "%s", error.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
