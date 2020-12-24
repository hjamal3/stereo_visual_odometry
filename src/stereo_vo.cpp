#include "stereo_visual_odometry/stereo_vo.h"

cv::Mat rosImage2CvMat(sensor_msgs::ImageConstPtr img) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
            cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
            return cv::Mat();
    }
    return cv_ptr->image;
}

void stereo_callback(const sensor_msgs::ImageConstPtr& image1, const sensor_msgs::ImageConstPtr& image2)
{

    static int count = 0;
    if (!count)
    {
        imageLeft_t0 = rosImage2CvMat(image1);
        imageRight_t0 = rosImage2CvMat(image2);
        count++;
        return;
    }

    // ------------
    // Load images
    // ------------
    imageLeft_t1 = rosImage2CvMat(image1);
    imageRight_t1 = rosImage2CvMat(image2);

    int frame_id = count;
    std::cout << std::endl << "frame id " << frame_id << std::endl;

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
        std::cout << "time features " << time_matching_features << std::endl;
        std::cout << "time total " << float(t_b - t_a)/CLOCKS_PER_SEC*1000 << std::endl;
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
    count++;
}


int main(int argc, char **argv)
{

    ros::init(argc, argv, "stereo_vo_node");

    ros::NodeHandle n;

    ros::Rate loop_rate(20);

    // using message_filters to get stereo callback on one topic
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, "left/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, "right/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image1_sub, image2_sub);
    sync.registerCallback(boost::bind(&stereo_callback, _1, _2));

    // #if USE_CUDA
    //     printf("CUDA Enabled\n");
    // #endif
    // -----------------------------------------
    // Load images and calibration parameters
    // -----------------------------------------
    // bool display_ground_truth = true;
    // std::vector<Matrix> pose_matrix_gt;
    // string filename_pose = "/home/hjamal/Desktop/Research/vo/dataset/sequences/00.txt";
    // pose_matrix_gt = loadPoses(filename_pose);

    // // Sequence hardcoded for now
    // string filepath = "/home/hjamal/Desktop/Research/vo/data_odometry_gray/dataset/sequences/00/";
    // cout << "Filepath: " << filepath << endl;

    // // Camera calibration
    // float fx = 721.5377;
    // float fy = 721.5377;
    // float cx = 609.5593;
    // float cy = 172.854;
    // float bf = -387.5744;
    
    // cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    // cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
    // cout << "K_left: " << endl << projMatrl << endl;
    // cout << "K_right: " << endl << projMatrr << endl;

    // // Initial variables
    // cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    // cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
    // cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);

    // std::cout << "frame_pose " << frame_pose << std::endl;
    // cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);
    // FeatureSet currentVOFeatures;

    // int init_frame_id = 0;
    // ------------------------
    // Load first images
    // ------------------------
    // cv::Mat imageRight_t0,  imageLeft_t0;
    // cv::Mat imageLeft_t0_color;
    // loadImageLeft(imageLeft_t0_color,  imageLeft_t0, init_frame_id, filepath);
    // cv::Mat imageRight_t0_color;  
    // loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath);
    // clock_t t_a, t_b;
    // clock_t t_1, t_2;

    int count = 0;
    while (ros::ok())
    {
        ros::spinOnce();

        // int frame_id = count;
        // std::cout << std::endl << "frame id " << frame_id << std::endl;
        // // ------------
        // // Load images
        // // ------------
        // cv::Mat imageRight_t1,  imageLeft_t1;

        // cv::Mat imageLeft_t1_color;
        // loadImageLeft(imageLeft_t1_color,  imageLeft_t1, frame_id, filepath);        
        // cv::Mat imageRight_t1_color;  
        // loadImageRight(imageRight_t1_color, imageRight_t1, frame_id, filepath);   

        // t_a = clock();
        // t_1 = clock();
        // std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;  
        // matchingFeatures( imageLeft_t0, imageRight_t0,
        //                   imageLeft_t1, imageRight_t1, 
        //                   currentVOFeatures,
        //                   pointsLeft_t0, 
        //                   pointsRight_t0, 
        //                   pointsLeft_t1, 
        //                   pointsRight_t1);  
        // t_2 = clock();
        // float time_matching_features = 1000*(double)(t_2-t_1)/CLOCKS_PER_SEC;

        // // set new images as old images
        // imageLeft_t0 = imageLeft_t1;
        // imageRight_t0 = imageRight_t1;

        // // ---------------------
        // // Triangulate 3D Points
        // // ---------------------
        // cv::Mat points3D_t0, points4D_t0;
        // cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
        // cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

        // // ---------------------
        // // Tracking transfomation
        // // ---------------------
        // // PnP
        // trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, points3D_t0, rotation, translation, false);
        // //displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);

        // // ------------------------------------------------
        // // Integrating and display
        // // ------------------------------------------------

        // cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
        // cv::Mat rigid_body_transformation;
        // if(abs(rotation_euler[1])<0.1 && abs(rotation_euler[0])<0.1 && abs(rotation_euler[2])<0.1)
        // {
        //     integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation);

        // } else {

        //     std::cout << "Too large rotation"  << std::endl;
        // }
        // t_b = clock();
        // float frame_time = 1000*(double)(t_b-t_a)/CLOCKS_PER_SEC;
        // float fps = 1000/frame_time;
        // //cout << "[Info] frame times (ms): " << frame_time << endl;
        // //cout << "[Info] FPS: " << fps << endl;
        // cv::Mat xyz = frame_pose.col(3).clone();
        // cv::Mat R = frame_pose(cv::Rect(0,0,3,3));

        // //display(frame_id, trajectory, xyz, pose_matrix_gt, fps, display_ground_truth);

        // // benchmark times
        // if (false)
        // {
        //     cout << "time features " << time_matching_features << std::endl;
        //     cout << "time total " << float(t_b - t_a)/CLOCKS_PER_SEC*1000 << std::endl;
        // }

        // // publish
        // if (true)
        // {
        //     std::cout << xyz << std::endl;
        //     static tf::TransformBroadcaster br;

        //     tf::Transform transform;
        //     transform.setOrigin( tf::Vector3(xyz.at<double>(0), xyz.at<double>(1), xyz.at<double>(2)) );
        //     tf::Quaternion q;
        //     tf::Matrix3x3 R_tf(R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),R.at<double>(1,0),
        //         R.at<double>(1,1),R.at<double>(1,2),R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2));
        //     R_tf.getRotation(q);
        //     transform.setRotation(q);
        //     br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "odom", "camera"));

        //     transform.setOrigin(tf::Vector3(0.0, 0.0,0.0));
        //     tf::Quaternion q2(0.5,-0.5,0.5,-0.5);
        //     transform.setRotation(q2);
        //     br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "odom"));
        // }

        loop_rate.sleep();
        ++count;
    }

    return 0;
}