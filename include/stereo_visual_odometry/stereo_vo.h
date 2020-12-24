#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include "feature.h"
#include "utils.h"
#include "visualOdometry.h"

cv::Mat imageRight_t0,  imageLeft_t0;
cv::Mat imageRight_t1,  imageLeft_t1;

// Camera calibration
float fx = 721.5377;
float fy = 721.5377;
float cx = 609.5593;
float cy = 172.854;
float bf = -387.5744;

cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
//std::cout << "K_left: " << endl << projMatrl << endl;
//std::cout << "K_right: " << endl << projMatrr << endl;

// Initial variables
cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);

// std::cout << "frame_pose " << frame_pose << std::endl;
cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);
FeatureSet currentVOFeatures;

clock_t t_a, t_b;
clock_t t_1, t_2;
