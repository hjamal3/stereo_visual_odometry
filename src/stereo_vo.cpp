#include "ros/ros.h"
#include "std_msgs/String.h"
#include <tf/transform_broadcaster.h>

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
#include "evaluate_odometry.h"
#include "visualOdometry.h"

using namespace std;

int main(int argc, char **argv)
{

    ros::init(argc, argv, "stereo_vo_node");

    ros::NodeHandle n;

    ros::Rate loop_rate(20);


    #if USE_CUDA
        printf("CUDA Enabled\n");
    #endif
    // -----------------------------------------
    // Load images and calibration parameters
    // -----------------------------------------
    bool display_ground_truth = true;
    std::vector<Matrix> pose_matrix_gt;
    string filename_pose = "/home/hjamal/Desktop/Research/vo/dataset/sequences/00.txt";
    pose_matrix_gt = loadPoses(filename_pose);

    // Sequence hardcoded for now
    string filepath = "/home/hjamal/Desktop/Research/vo/data_odometry_gray/dataset/sequences/00/";
    cout << "Filepath: " << filepath << endl;

    // Camera calibration
    float fx = 718.8560;
    float fy = 718.8560;
    float cx = 607.1928;
    float cy = 185.2157;
    float bf = -386.1448;
    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
    cout << "K_left: " << endl << projMatrl << endl;
    cout << "K_right: " << endl << projMatrr << endl;

    // Initial variables
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);

    std::cout << "frame_pose " << frame_pose << std::endl;
    cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);
    FeatureSet currentVOFeatures;

    int init_frame_id = 0;

    // ------------------------
    // Load first images
    // ------------------------
    cv::Mat imageRight_t0,  imageLeft_t0;
    cv::Mat imageLeft_t0_color;
    loadImageLeft(imageLeft_t0_color,  imageLeft_t0, init_frame_id, filepath);
    cv::Mat imageRight_t0_color;  
    loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath);
    clock_t t_a, t_b;
    clock_t t_1, t_2;

    int count = 0;
    while (ros::ok())
    {
        ros::spinOnce();

        int frame_id = count;
        std::cout << std::endl << "frame id " << frame_id << std::endl;
        // ------------
        // Load images
        // ------------
        cv::Mat imageRight_t1,  imageLeft_t1;

        cv::Mat imageLeft_t1_color;
        loadImageLeft(imageLeft_t1_color,  imageLeft_t1, frame_id, filepath);        
        cv::Mat imageRight_t1_color;  
        loadImageRight(imageRight_t1_color, imageRight_t1, frame_id, filepath);   

        t_a = clock();
        t_1 = clock();
        std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;  
        matchingFeatures( imageLeft_t0, imageRight_t0,
                          imageLeft_t1, imageRight_t1, 
                          currentVOFeatures,
                          pointsLeft_t0, 
                          pointsRight_t0, 
                          pointsLeft_t1, 
                          pointsRight_t1);  
        t_2 = clock();
        float time_matching_features = 1000*(double)(t_2-t_1)/CLOCKS_PER_SEC;

        // set new images as old images
        imageLeft_t0 = imageLeft_t1;
        imageRight_t0 = imageRight_t1;

        // ---------------------
        // Triangulate 3D Points
        // ---------------------
        cv::Mat points3D_t0, points4D_t0;
        cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
        cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

        // ---------------------
        // Tracking transfomation
        // ---------------------
        // PnP
        trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, points3D_t0, rotation, translation, false);
        //displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);

        // ------------------------------------------------
        // Integrating and display
        // ------------------------------------------------

        cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
        cv::Mat rigid_body_transformation;
        if(abs(rotation_euler[1])<0.1 && abs(rotation_euler[0])<0.1 && abs(rotation_euler[2])<0.1)
        {
            integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation);

        } else {

            std::cout << "Too large rotation"  << std::endl;
        }
        t_b = clock();
        float frame_time = 1000*(double)(t_b-t_a)/CLOCKS_PER_SEC;
        float fps = 1000/frame_time;
        //cout << "[Info] frame times (ms): " << frame_time << endl;
        //cout << "[Info] FPS: " << fps << endl;
        cv::Mat xyz = frame_pose.col(3).clone();
        cv::Mat R = frame_pose(cv::Rect(0,0,3,3));

        //display(frame_id, trajectory, xyz, pose_matrix_gt, fps, display_ground_truth);

        // benchmark times
        if (false)
        {
            cout << "time features " << time_matching_features << std::endl;
            cout << "time total " << float(t_b - t_a)/CLOCKS_PER_SEC*1000 << std::endl;
        }

        // publish
        if (true)
        {
            std::cout << xyz << std::endl;
            static tf::TransformBroadcaster br;

            tf::Transform transform;
            transform.setOrigin( tf::Vector3(xyz.at<double>(0), xyz.at<double>(1), xyz.at<double>(2)) );
            tf::Quaternion q;
            tf::Matrix3x3 R_tf(R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),R.at<double>(1,0),
                R.at<double>(1,1),R.at<double>(1,2),R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2));
            R_tf.getRotation(q);
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "odom", "camera"));

            transform.setOrigin(tf::Vector3(0.0, 0.0,0.0));
            tf::Quaternion q2(0.5,-0.5,0.5,-0.5);
            transform.setRotation(q2);
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "odom"));
        }

        loop_rate.sleep();
        ++count;
    }

    return 0;
}