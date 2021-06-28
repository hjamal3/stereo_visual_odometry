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

// How much the wheel encoder model translated
Eigen::Matrix<double,3,1> encoders_translation(3);
Eigen::Matrix<double,3,1> vo_translation(3);
Eigen::Matrix<double,3,1> global_pos(3);
Eigen::Quaternion<double> vo_rot;
Eigen::Quaternion<double> current_rot;

// camera to imu frame, wxyz

// Note: eigen is w x y z and tf is x y z w
class PoseEstimator
{
	public:

		PoseEstimator(cv::Mat projMatrl_, cv::Mat projMatrr_);

		cv::Mat rosImage2CvMat(const sensor_msgs::ImageConstPtr img);

		// stereo pair callback
		void stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right);

		// orientation callback
		void quat_callback(const::geometry_msgs::Quaternion::ConstPtr& msg);

		// wheel encoding callback
		void encoders_callback(const std_msgs::Int32MultiArray::ConstPtr& msg);

		// greyscale conversion
		void to_greyscale(const cv::Mat &img_color, cv::Mat &img_grey);

		// runs the pipeline
		void run();

		// logging 
		bool logging_path;
		std::ofstream output_file;

		// mode of operation
		bool use_vo;

	private:

		int frame_id = 0;

		// projection matrices for camera
		cv::Mat projMatrl, projMatrr;

		// images of current and next time step
		cv::Mat imageRight_t0,  imageLeft_t0;
		cv::Mat imageRight_t1,  imageLeft_t1;

		// number of features sufficient for VO
		const int features_threshold = 60;

		// set of features currently tracked
		FeatureSet currentVOFeatures;

		// initial pose variables
		cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
		cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
		cv::Mat rotation_rodrigues = cv::Mat::zeros(3, 1, CV_64F);

		// for post processing
		// cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);

		// wheel encoders
		bool first_time_enc = true;
		bool orientation_init = false;
		int ticks_l_prev = 0;
		int ticks_r_prev = 0;
		const double R = 0.1016;
		const double ticks_per_m = 1316/(M_PI*2*R);
		// Camera body transformation, wxyz
		const Eigen::Quaternion<double> q_bc = {0.3903,-0.5896,0.5896,-0.3903};	

};
