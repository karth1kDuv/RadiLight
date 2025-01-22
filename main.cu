// CUDA and OpenCV includes
#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "nvapriltags/include/nvAprilTags.h"

// Standard library and utility includes
#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>


struct AprilTagsImpl {
    nvAprilTagsHandle april_tags_handle = nullptr;
    nvAprilTagsCameraIntrinsics_t cam_intrinsics;
    std::vector<nvAprilTagsID_t> tags;
    cudaStream_t main_stream = {};
    nvAprilTagsImageInput_t input_image;
    uchar4 *input_image_buffer = nullptr;
    size_t input_image_buffer_size = 0;
    int max_tags;

    void initialize(uint32_t width, uint32_t height, size_t image_buffer_size, size_t pitch_bytes,
                    float tag_edge_size_, int max_tags_) {
        assert(!april_tags_handle && "Already initialized.");
        cam_intrinsics = {802.9265702293416, 803.6309422064642, 966.4221440154661, 477.6409613889424};

        const int error = nvCreateAprilTagsDetector(
            &april_tags_handle, width, height, nvAprilTagsFamily::NVAT_TAG36H11,
            &cam_intrinsics, tag_edge_size_);

        if (error != 0) {
            throw std::runtime_error("Failed to create NV April Tags detector (error code " +
                                     std::to_string(error) + ")");
        }

        cudaStreamCreate(&main_stream);
        tags.resize(max_tags_);
        max_tags = max_tags_;

        if (input_image_buffer != nullptr) {
            cudaFree(input_image_buffer);
        }

        const cudaError_t cuda_error = cudaHostAlloc((void**)&input_image_buffer, image_buffer_size, cudaHostAllocDefault);
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error("Could not allocate CUDA memory (error code " +
                                     std::to_string(cuda_error) + ")");
        }

        input_image_buffer_size = image_buffer_size;
        input_image.width = width;
        input_image.height = height;
        input_image.dev_ptr = reinterpret_cast<uchar4 *>(input_image_buffer);
        input_image.pitch = pitch_bytes;
    }

    ~AprilTagsImpl() {
        if (april_tags_handle != nullptr) {
            cudaStreamDestroy(main_stream);
            nvAprilTagsDestroy(april_tags_handle);
            cudaFree(input_image_buffer);
        }
    }
};

void roborioSender(const std::vector<std::array<float, 8>> &sendData, int sock) {
    if (sendData.empty()) {
        std::cout << "No data to send." << std::endl;
        return;
    }

    sockaddr_in destination = {AF_INET, htons(9000), inet_addr("192.168.86.239")};
    size_t size = sendData.size() * sizeof(std::array<float, 8>);

    int n_bytes = ::sendto(sock, sendData.data(), size, 0, 
                           reinterpret_cast<sockaddr *>(&destination), sizeof(destination));

    if (n_bytes < 0) {
        std::cerr << "Failed to send data: " << strerror(errno) << std::endl;
    } else {
        std::cout << "Sent " << n_bytes << " bytes (" << sendData.size() << " tags)" << std::endl;
    }
}

std::array<float, 4> eulerToQuaternion(double roll, double pitch, double yaw) {
    std::array<float, 4> quaternion;
    double halfRoll = roll / 2.0, halfPitch = pitch / 2.0, halfYaw = yaw / 2.0;
    quaternion[0] = cos(halfYaw) * cos(halfPitch) * cos(halfRoll) + sin(halfYaw) * sin(halfPitch) * sin(halfRoll);
    quaternion[1] = cos(halfYaw) * cos(halfPitch) * sin(halfRoll) - sin(halfYaw) * sin(halfPitch) * cos(halfRoll);
    quaternion[2] = sin(halfYaw) * cos(halfPitch) * cos(halfRoll) + cos(halfYaw) * sin(halfPitch) * sin(halfRoll);
    quaternion[3] = sin(halfYaw) * cos(halfPitch) * cos(halfRoll) - cos(halfYaw) * sin(halfPitch) * sin(halfRoll);
    return quaternion;
}

std::array<float, 8> getPose(std::vector<cv::Point2d> imagePts, int TagId){
    static std::array<float, 8> lastValidPose = {0}; 
    std::array<float, 8> finalPose = {0};

    if (imagePts.empty()) {
        std::cout << "No detections" << "\n";
        return lastValidPose;
    }

    std::vector<cv::Point3f> cvApriltagPts = 
    {cv::Point3f(-0.08255, 0.08255, 0),
    cv::Point3f(0.08255, 0.08255, 0),
    cv::Point3f(0.08255, -0.08255, 0),
    cv::Point3f(-0.08255, -0.08255, 0)};

    float K[] = {802.9265702293416f, 0, 966.4221440154661f, 
                0, 803.6309422064642f, 477.6409613889424f, 
                0, 0, 1};

    cv::Mat cameraMatrix(3, 3, CV_32F, K);

    float distCoeffs[5] = {0}; 
    cv::Mat distortionMatrix(1, 5, CV_32F, distCoeffs);

    cv::Mat rvec, tvec;
    bool success = cv::solvePnPRansac(cvApriltagPts, imagePts, cameraMatrix, distortionMatrix, rvec, tvec, false, 10000000000, 4.0f, .99, cv::noArray(), cv::SOLVEPNP_IPPE_SQUARE);
    cv::solvePnPRefineLM(cvApriltagPts, imagePts, cameraMatrix, distortionMatrix, rvec, tvec);
    if (!success){
    std::cerr << "Error with SOLVEPNP" << "\n";
        return lastValidPose;
    };
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    R = R.t();  
    cv::Mat tvecI = -R * tvec;

    finalPose[0] = +tvecI.at<double>(2, 0);
    finalPose[1] = -tvecI.at<double>(0, 0);
    finalPose[2] = -tvecI.at<double>(1, 0);

    double roll = +rvec.at<double>(2, 0);
    double pitch = -rvec.at<double>(0, 0);
    double yaw = +rvec.at<double>(1, 0);
    
    std::array<float, 4> quaternion = eulerToQuaternion(roll, pitch, yaw);

    finalPose[3] = quaternion[0];
    finalPose[4] = quaternion[1];
    finalPose[5] = quaternion[2];
    finalPose[6] = quaternion[3];
    finalPose[7] = TagId;

    lastValidPose = finalPose;
    return finalPose;
}


std::atomic<bool> processing_done(true);
std::mutex frame_mutex;
cv::VideoCapture capture;
cv::Mat frame, img_rgba8;

void captureThread() {
    while (capture.isOpened()) {
        cv::Mat new_frame;
        capture >> new_frame;
        if (new_frame.empty()) {
            break;
        }
        std::lock_guard<std::mutex> lock(frame_mutex);
        frame = new_frame;
        processing_done = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void processingThread(AprilTagsImpl *impl_, int roborioSock) { 
    std::vector<cv::Point2d> imagePts;
    std::vector<std::array<float, 8>> finalToSend;

    while (capture.isOpened()) {
        auto start = std::chrono::steady_clock::now(); 

        if (!processing_done) {
            cv::Mat local_frame;
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                if (frame.empty()) {
                    processing_done = true;
                    continue;
                }
                frame.copyTo(local_frame);
                //cv::resize(local_frame, local_frame, cv::Size(local_frame.cols / 2, local_frame.rows / 2), 0, 0, cv::INTER_LINEAR);
                cv::cvtColor(local_frame, img_rgba8, cv::COLOR_BGR2RGBA);
            }
            auto frame_end = std::chrono::steady_clock::now();
            
            auto memory_start = std::chrono::steady_clock::now();

            const cudaError_t cuda_error =
                cudaMemcpyAsync(impl_->input_image_buffer, (uchar4 *)img_rgba8.ptr<unsigned char>(0),
                impl_->input_image_buffer_size, cudaMemcpyHostToDevice, impl_->main_stream);

            if (cuda_error != cudaSuccess) {
                throw std::runtime_error(
                        "Could not memcpy to device CUDA memory (error code " +
                        std::to_string(cuda_error) + ")");
            }
            auto memory_end = std::chrono::steady_clock::now();
            
            
            auto detect_start = std::chrono::steady_clock::now();
            uint32_t numDetections;
            const int error = nvAprilTagsDetect(
                    impl_->april_tags_handle, &(impl_->input_image), impl_->tags.data(),
                    &numDetections, impl_->max_tags, impl_->main_stream);
            if (error != 0) {
                throw std::runtime_error("Failed to run AprilTags detector (error code " +
                                        std::to_string(error) + ")");
            }

            

            auto detect_end = std::chrono::steady_clock::now();

            
            auto getPose_start = std::chrono::steady_clock::now();

            std::cout << numDetections;
            std::cout << std::endl;

            for (int i = 0; i < numDetections; i++) {
                const nvAprilTagsID_t &detection = impl_->tags[i];
                for (auto corner : detection.corners) {
                    imagePts.push_back(cv::Point2d(corner.x, corner.y));
                    cv::circle(frame, cv::Point(corner.x, corner.y), 3, cv::Scalar(255, 255, 255));
                }
                finalToSend.push_back(getPose(imagePts, detection.id));
                imagePts.clear();
            }

            for (const auto &pose : finalToSend) {
                std::cout << "Tag ID: " << pose[7] << " | Pose: ";
                for (size_t i = 0; i < 7; i++) {
                    std::cout << pose[i] << " ";
                }
                std::cout << std::endl;
            }
            
            roborioSender(finalToSend, roborioSock);
            auto getPose_end = std::chrono::steady_clock::now();

            // Total time for frame processing
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> frame_time = end - start;
            
            std::chrono::duration<double> get_frame_time = frame_end - start;
            std::chrono::duration<double> memory_time = memory_end - memory_start;
            std::chrono::duration<double> detect_time = detect_end - detect_start;
            std::chrono::duration<double> pose_time = getPose_end - getPose_start;

            std::cout << "FrameTime: " << get_frame_time.count() << " s\n";
            std::cout << "MemoryTime: " << memory_time.count() << " s\n";
            std::cout << "DetectTime: " << detect_time.count() << " s\n";
            std::cout << "PoseTime: " << pose_time.count() << " s\n";
            std::cout << "FPS: " << 1.0 / frame_time.count() << "\n";
            std::cout << "\n";

            finalToSend.clear();
            cv::imshow("frame", frame); 
            if (cv::waitKey(10)==27)
                break;
            processing_done = true;
        }
    }
}

#include "ntcore.h"

int main() {
    NT_Inst inst = nt::GetDefaultInstance(); // This should now work!!

    int roborioSock = ::socket(AF_INET, SOCK_DGRAM, 0);
    capture.open(0, cv::CAP_V4L2);
    capture.set(cv::CAP_PROP_FPS, 90);
    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1260);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 980);
    capture >> frame;
    //cv::resize(frame, frame, cv::Size(frame.cols / 2, frame.rows / 2), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(frame, img_rgba8, cv::COLOR_BGR2RGBA);
    auto *impl_ = new AprilTagsImpl();
    impl_->initialize(img_rgba8.cols, img_rgba8.rows,
                      img_rgba8.total() * img_rgba8.elemSize(),  img_rgba8.step, .1651f, 4);

    std::thread cThread(captureThread);
    std::thread pThread(processingThread, impl_, roborioSock);

    cThread.join();
    pThread.join();
    capture.release();
    ::close(roborioSock);
    delete impl_;
    return 0;
}
