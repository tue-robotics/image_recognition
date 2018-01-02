#include <ros/ros.h>
#include <jetson-inference/detectNet.h>

#include <jetson-inference/cudaMappedMemory.h>
#include <jetson-inference/loadImage.h>
#include <jetson-inference/cudaFont.h>

#include <opencv2/core.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <image_recognition_msgs/Recognitions.h>

class DetectNetROS
{
  public:
    DetectNetROS()
    {
      // get node handles
      ros::NodeHandle private_nh("~");
      ros::NodeHandle nh;

      // get parameters from server, checking for errors as it goes
      std::string prototxt_path, model_path;
      if (!private_nh.getParam("prototxt_path", prototxt_path))
      {
        ROS_ERROR("unable to read ~prototxt_path for imagenet_ node");
        exit(1);
      }
      if (!private_nh.getParam("model_path", model_path))
      {
        ROS_ERROR("unable to read ~model_path for imagenet_ node");
        exit(1);
      }

      // make sure files exist (and we can read them)
      if( access(prototxt_path.c_str(), R_OK) )
      {
        ROS_ERROR("unable to read file \"%s\", check filename and permissions",prototxt_path.c_str());
        exit(1);
      }
      if( access(model_path.c_str(), R_OK) ) 
      {
        ROS_ERROR("unable to read file \"%s\", check filename and permissions",model_path.c_str());
        exit(1);
      }

      // create imageNet
      net_ = detectNet::Create(prototxt_path.c_str(), 
                               model_path.c_str(), 
                               private_nh.param("mean_pixel", 117.0),
                               private_nh.param("threshold", 0.5));

      if( !net_ )
      {
        ROS_INFO("Failed to initialize detectNet");
        return;
      }

      /*
       * allocate memory for output bounding boxes and class confidence
       */
      max_boxes_ = net_->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", max_boxes_);
      classes_  = net_->GetNumClasses();
      
      bounding_box_CPU_    = NULL;
      bounding_box_CUDA_   = NULL;
      confidence_CPU_  = NULL;
      confidence_CUDA_ = NULL;
      
      if( !cudaAllocMapped((void**)&bounding_box_CPU_, (void**)&bounding_box_CUDA_, max_boxes_ * sizeof(float4)) ||
          !cudaAllocMapped((void**)&confidence_CPU_, (void**)&confidence_CUDA_, max_boxes_ * classes_ * sizeof(float)) )
      {
      	ROS_ERROR("failed to alloc output memory");
      }

      // setup image transport
      image_transport::ImageTransport it(private_nh);

      // subscriber for passing in images
      image_subscriber_ = it.subscribe("imin", 10, &DetectNetROS::callback, this);

      recognitions_publisher_ = nh.advertise<image_recognition_msgs::Recognitions>("recognitions", 1);

      // init gpu memory
      gpu_data_ = NULL;

      ROS_INFO("DetectNetROS initialized ...");
    }

    ~DetectNetROS()
    {
      ROS_INFO("Shutting down...");
      if(gpu_data_)
        CUDA(cudaFree(gpu_data_));
      delete net_;
    }


  private:
    void callback(const sensor_msgs::ImageConstPtr& input)
    {
      cv::Mat cv_im = cv_bridge::toCvCopy(input, "bgr8")->image;

      // convert bit depth
      cv_im.convertTo(cv_im, CV_32FC3);

      // convert color
      cv::cvtColor(cv_im, cv_im, CV_BGR2RGBA);

      // allocate GPU data if necessary
      if (!gpu_data_)
      {
        ROS_INFO("first allocation");
        CUDA(cudaMalloc(&gpu_data_, cv_im.rows*cv_im.cols * sizeof(float4)));
      } else if(image_height_ != cv_im.rows || image_width_ != cv_im.cols){
        ROS_INFO("re allocation");
        // reallocate for a new image size if necessary
        CUDA(cudaFree(gpu_data_));
        CUDA(cudaMalloc(&gpu_data_, cv_im.rows*cv_im.cols * sizeof(float4)));
      }

      image_height_ = cv_im.rows;
      image_width_ = cv_im.cols;
      image_size_ = cv_im.rows*cv_im.cols * sizeof(float4);
      float4* cpu_data = (float4*)(cv_im.data);

      // copy to device
      CUDA(cudaMemcpy(gpu_data_, cpu_data, image_size_, cudaMemcpyHostToDevice));

      int number_of_bounding_boxes_ = max_boxes_;

      image_recognition_msgs::Recognitions msg;
      msg.header = input->header;
      if (net_->Detect((float*)gpu_data_, image_width_, image_height_, bounding_box_CPU_, &number_of_bounding_boxes_, confidence_CPU_)) {
        ROS_INFO("Detected %d bounding boxes", number_of_bounding_boxes_);
        msg.recognitions.resize(number_of_bounding_boxes_);
		
        for(int n=0; n < number_of_bounding_boxes_; n++)
        {
	  const int class_confidence = confidence_CPU_[n*2+1];
          float* bounding_box = bounding_box_CPU_ + (n * 4);
          msg.recognitions[n].roi.x_offset = bounding_box[0];
          msg.recognitions[n].roi.y_offset = bounding_box[1];
          msg.recognitions[n].roi.width = bounding_box[2] - bounding_box[0];
          msg.recognitions[n].roi.height = bounding_box[3] - bounding_box[1];
          msg.recognitions[n].categorical_distribution.probabilities.resize(1);
          msg.recognitions[n].categorical_distribution.probabilities[0].label = "TODO";
          msg.recognitions[n].categorical_distribution.probabilities[0].probability = 1.0;
        }
      }

      recognitions_publisher_.publish(msg);
    }

    // private variables
    image_transport::Subscriber image_subscriber_;
    detectNet* net_;

    ros::Publisher recognitions_publisher_;

    float4* gpu_data_;
    float* bounding_box_CPU_;
    float* bounding_box_CUDA_;
    float* confidence_CPU_;
    float* confidence_CUDA_;
 
    uint32_t max_boxes_;
    uint32_t classes_;

    uint32_t image_width_;
    uint32_t image_height_;
    size_t   image_size_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "detect_net");
  DetectNetROS net;
  ros::spin();
  return 0;
}
