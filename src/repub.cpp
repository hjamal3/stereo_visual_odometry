// One time helper code... 

#include "ros/ros.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "sensor_msgs/Image.h"

ros::Publisher left_pub;
ros::Publisher right_pub;

cv::Mat rosImage2CvMat(sensor_msgs::ImageConstPtr img) {
    cv_bridge::CvImagePtr cv_ptr;
    try 
    {
		cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
		std::cout << "exception" << std::endl;
		return cv::Mat();
    }
    return cv_ptr->image;
}

void stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right)
{
	cv::Mat left = rosImage2CvMat(image_left);
	cv::Mat right = rosImage2CvMat(image_right);
	cv::cvtColor(left, left, CV_RGB2GRAY);
	cv::cvtColor(right, right, CV_RGB2GRAY);
	cv_bridge::CvImage out_msg0;
	cv_bridge::CvImage out_msg1;
	out_msg0.encoding = sensor_msgs::image_encodings::MONO8;
	out_msg1.encoding = sensor_msgs::image_encodings::MONO8;
	out_msg0.image = left;
	out_msg0.header.frame_id = "/stereo/left";
	out_msg0.header.stamp = image_left->header.stamp;
	out_msg1.image = right;
	out_msg1.header.frame_id = "/stereo/right";
	out_msg1.header.stamp = image_right->header.stamp;//time1;
	left_pub.publish(out_msg0.toImageMsg());
	right_pub.publish(out_msg1.toImageMsg());
}


int main(int argc, char **argv)
{

    ros::init(argc, argv, "repub_node");

    ros::NodeHandle n;

    // using message_filters to get stereo callback on one topic
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, "/stereo/left/image_rect_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, "/stereo/right/image_rect_color", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(1), image1_sub, image2_sub);
    sync.registerCallback(&stereo_callback);

    std::cout << "Repub Node Initialized!" << std::endl;
    
	left_pub = n.advertise<sensor_msgs::Image>("/stereo/left/image_rect", 0);
	right_pub = n.advertise<sensor_msgs::Image>("/stereo/right/image_rect", 0);

    ros::spin();
    return 0;
}
