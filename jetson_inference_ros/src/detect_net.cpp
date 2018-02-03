#include "detect_net.h"

#include <jetson-inference/detectNet.h>

#include <jetson-inference/cudaMappedMemory.h>
#include <jetson-inference/loadImage.h>
#include <jetson-inference/cudaFont.h>

detectNet* g_net = NULL;
float4* g_gpu_data = NULL;
float* g_bounding_box_CPU = NULL;
float* g_bounding_box_CUDA = NULL;
float* g_confidence_CPU = NULL;
float* g_confidence_CUDA = NULL;
uint32_t g_max_boxes;
uint32_t g_classes;
uint32_t g_image_width;
uint32_t g_image_height;
size_t   g_image_size;

DetectNet::DetectNet(std::string prototxt_path, std::string model_path, int mean_pixel, float threshold)
{
  // make sure files exist (and we can read them)
  if( access(prototxt_path.c_str(), R_OK) )
  {
    throw std::runtime_error("Unable to read prototxt file: " + prototxt_path);
  }
  if( access(model_path.c_str(), R_OK) )
  {
    throw std::runtime_error("Unable to read model path file: " + model_path);
  }

  // create imageNet
  g_net = detectNet::Create(prototxt_path.c_str(), model_path.c_str(), mean_pixel, threshold);

  if( !g_net )
  {
    throw std::runtime_error("Failed to initialize detectNet");
  }

  /*
   * allocate memory for output bounding boxes and class confidence
   */
  g_max_boxes = g_net->GetMaxBoundingBoxes();
  g_classes  = g_net->GetNumClasses();

  if( !cudaAllocMapped((void**)&g_bounding_box_CPU, (void**)&g_bounding_box_CUDA, g_max_boxes * sizeof(float4)) ||
      !cudaAllocMapped((void**)&g_confidence_CPU, (void**)&g_confidence_CUDA, g_max_boxes * g_classes * sizeof(float)) )
  {
    throw std::runtime_error("Failed to alloc output memory");
  }
}

DetectNet::~DetectNet()
{
  if(g_gpu_data)
    CUDA(cudaFree(g_gpu_data));
  delete g_net;
}

std::vector<image_recognition_msgs::Recognition> DetectNet::processImage(const cv::Mat &cv_im)
{
  // allocate GPU data if necessary
  if (!g_gpu_data)
  {
    CUDA(cudaMalloc(&g_gpu_data, cv_im.rows * cv_im.cols * sizeof(float4)));
  } 
  else if (g_image_height != cv_im.rows || g_image_width != cv_im.cols)
  {
    CUDA(cudaFree(g_gpu_data));
    CUDA(cudaMalloc(&g_gpu_data, cv_im.rows*cv_im.cols * sizeof(float4)));
  }

  g_image_height = cv_im.rows;
  g_image_width = cv_im.cols;
  g_image_size = cv_im.rows * cv_im.cols * sizeof(float4);
  float4* cpu_data = (float4*)(cv_im.data);

  // copy to device
  CUDA(cudaMemcpy(g_gpu_data, cpu_data, g_image_size, cudaMemcpyHostToDevice));

  int number_of_bounding_boxes = g_max_boxes;

  std::vector<image_recognition_msgs::Recognition> recognitions;
  if (g_net->Detect((float*)g_gpu_data, g_image_width, g_image_height, g_bounding_box_CPU, &number_of_bounding_boxes, g_confidence_CPU)) 
  {
    recognitions.resize(number_of_bounding_boxes);

    for(int n = 0; n < number_of_bounding_boxes; n++)
    {
      // confidence optional pointer to float2 array filled with a (confidence, class) pair for each bounding box (numBoxes)
      const float confidence = g_confidence_CPU[n*2];
      const int label = g_confidence_CPU[n*2+1];
      float* bounding_box = g_bounding_box_CPU + (n * 4);
      recognitions[n].roi.x_offset = bounding_box[0];
      recognitions[n].roi.y_offset = bounding_box[1];
      recognitions[n].roi.width = bounding_box[2] - bounding_box[0];
      recognitions[n].roi.height = bounding_box[3] - bounding_box[1];
      recognitions[n].categorical_distribution.probabilities.resize(1);
      recognitions[n].categorical_distribution.probabilities[0].label = std::to_string(label);
      recognitions[n].categorical_distribution.probabilities[0].probability = confidence;
    }
  }

  return recognitions;
}
