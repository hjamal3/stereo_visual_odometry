#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/Int32MultiArray.h"
#include "geometry_msgs/Quaternion.h"
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

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry> 


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
#include "vo.h"
#include <math.h>

// wheel encoders
bool first_time_enc = true;
bool first_time_quat = true;
int ticks_l_prev = 0;
int ticks_r_prev = 0;
const double L = 0.5842;
const double R = 0.1016;
const double ticks_per_m = 1316/(M_PI*2*R);
Eigen::Matrix<double,3,1> encoders_translation(3);
Eigen::Matrix<double,3,1> vo_translation(3);
Eigen::Quaternion<double> vo_rot;
Eigen::Matrix<double,3,1> global_pos(3);
Eigen::Quaternion<double> current_rot;
const Eigen::Quaternion<double> R_bc(0.3799673, -0.5963445, 0.5963423, -0.3799659); // camera to body frame
// Note: eigen is w x y z and tf is x y z w

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
