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

class StereoVO
{
	public:

		StereoVO(cv::Mat projMatrl_, cv::Mat projMatrr_);

		cv::Mat rosImage2CvMat(sensor_msgs::ImageConstPtr img);

		// stereo pair callback
		void stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right);

		// runs the pipeline
		void run();

	private:

		int frame_id = 0;

		// projection matrices for camera
		cv::Mat projMatrl, projMatrr;

		// images of current and next time step
		cv::Mat imageRight_t0,  imageLeft_t0;
		cv::Mat imageRight_t1,  imageLeft_t1;

		// initial pose variables
		cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
		cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);

		// std::cout << "frame_pose " << frame_pose << std::endl;
		cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);

		// set of features currently tracked
		FeatureSet currentVOFeatures;

		// for timing code
		clock_t t_a, t_b;
		clock_t t_1, t_2;
};
