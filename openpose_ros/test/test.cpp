// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

int openPoseTutorialPose1()
{
    op::log("OpenPose Library Tutorial - Example 1.", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
        // - 0 will output all the logging messages
        // - 255 will output nothing
    // Step 2 - Read Google flags (user defined configuration)
    // outputSize
    const op::Point<int> outputSize(1280, 720);
    // netInputSize
    const op::Point<int> netInputSize(368, 368);
    // netOutputSize
    const auto netOutputSize = netInputSize;
    // poseModel
    const auto poseModel = op::flagsToPoseModel("COCO");

    // Step 3 - Initialize all required classes
    op::CvMatToOpInput cvMatToOpInput{netInputSize, 1, (float) 0.3};
    op::CvMatToOpOutput cvMatToOpOutput{outputSize};
    op::PoseExtractorCaffe poseExtractorCaffe{netInputSize, netOutputSize, outputSize, 1, poseModel,
                                              "/home/ubuntu/openpose/models/", 0};
    op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)0.05,
                                  true, (float)0.6};
    op::OpOutputToCvMat opOutputToCvMat{outputSize};
    const op::Point<int> windowedSize = outputSize;
    op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose Tutorial - Example 1"};
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorCaffe.initializationOnThread();
    poseRenderer.initializationOnThread();

    // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    cv::Mat inputImage = op::loadImage("/home/ubuntu/openpose/examples/media/COCO_val2014_000000000192.jpg", CV_LOAD_IMAGE_COLOR); // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    // Step 2 - Format input image to OpenPose input and output formats
    op::Array<float> netInputArray;
    std::vector<float> scaleRatios;
    std::tie(netInputArray, scaleRatios) = cvMatToOpInput.format(inputImage);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);
    // Step 3 - Estimate poseKeypoints
    poseExtractorCaffe.forwardPass(netInputArray, {inputImage.cols, inputImage.rows}, scaleRatios);
    const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();
    // Step 4 - Render poseKeypoints
    poseRenderer.renderPose(outputArray, poseKeypoints);
    // Step 5 - OpenPose output format to cv::Mat
    auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

    // ------------------------- SHOWING RESULT AND CLOSING -------------------------
    // Step 1 - Show results
    //frameDisplayer.displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
    // Step 2 - Logging information message
    op::log("Example 1 successfully finished.", op::Priority::High);
    // Return successful message
    return 0;
}

int main(int argc, char *argv[])
{
    // Running openPoseTutorialPose1
    return openPoseTutorialPose1();
}
