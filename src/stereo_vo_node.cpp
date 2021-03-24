#include "stereo_vo_node.h"
#include <stdexcept>

StereoVO::StereoVO(cv::Mat projMatrl_, cv::Mat projMatrr_)
{
    projMatrl = projMatrl_;
    projMatrr = projMatrr_;
}

cv::Mat StereoVO::rosImage2CvMat(sensor_msgs::ImageConstPtr img) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
            cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
            return cv::Mat();
    }
    return cv_ptr->image;
}

void StereoVO::stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right)
{

    if (!frame_id)
    {
        imageLeft_t0 = rosImage2CvMat(image_left);
        imageRight_t0 = rosImage2CvMat(image_right);
        frame_id++;
        return;
    }

    imageLeft_t1 = rosImage2CvMat(image_left);
    imageRight_t1 = rosImage2CvMat(image_right);

    // run the pipeline
    run();
}

void StereoVO::run()
{
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

    // set new images as old images
    imageLeft_t0 = imageLeft_t1;
    imageRight_t0 = imageRight_t1;

    // display visualize feature tracks
    displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);

    if (currentVOFeatures.size() < 30 ) //TODO should this be AND?
    {
        std::cout << "not enough features matched for pose estimation" << std::endl;
        frame_id++;
        return;  
    }

    // ---------------------
    // Triangulate 3D Points
    // ---------------------
    cv::Mat points3D_t0, points4D_t0;
    cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
    cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

    // ---------------------
    // Tracking transfomation
    // ---------------------
    // PnP: computes rotation and translation between pair of images
    trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t1, points3D_t0, rotation, translation, false);

    // ------------------------------------------------
    // Integrating
    // ------------------------------------------------
    cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
    if(abs(rotation_euler[1])<0.2 && abs(rotation_euler[0])<0.2 && abs(rotation_euler[2])<0.2)
    {
        integrateOdometryStereo(frame_id, frame_pose, rotation, translation);

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

    // benchmark times
    if (false)
    {
        float time_matching_features = 1000*(double)(t_2-t_1)/CLOCKS_PER_SEC;
        std::cout << "time to match features " << time_matching_features << std::endl;
        std::cout << "time total " << float(t_b - t_a)/CLOCKS_PER_SEC*1000 << std::endl;
    }

    // publish
    if (true)
    {
        std::cout << xyz << std::endl;
        static tf::TransformBroadcaster br;

	// transform of robot
        tf::Transform transform;
        transform.setOrigin( tf::Vector3(xyz.at<double>(0), xyz.at<double>(1), xyz.at<double>(2)) );
        tf::Quaternion q;
        tf::Matrix3x3 R_tf(R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),R.at<double>(1,0),
            R.at<double>(1,1),R.at<double>(1,2),R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2));
        R_tf.getRotation(q);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "vo_start", "camera"));

        transform.setOrigin(tf::Vector3(0.0, 0.0,0.0));
        tf::Quaternion q2(-0.6484585017186756, 0.648460883644128, -0.28195800926029446, 0.2819569735725981);
        transform.setRotation(q2);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "vo_start"));
    }
    frame_id++;
}


int main(int argc, char **argv)
{

    ros::init(argc, argv, "stereo_vo_node");

    ros::NodeHandle n;

    ros::Rate loop_rate(20);

    std::string filename; //TODO correct the name
    if (!(n.getParam("calib_yaml",filename)))
    {
        std::cerr << "no calib yaml" << std::endl;
        throw;
    }
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(!(fs.isOpened()))
    {
        std::cerr << "cv failed to load yaml" << std::endl;
        throw;
    }
    float fx, fy, cx, cy, bf; // Projection matrix parameters
    fs["fx"] >> fx;
    fs["fy"] >> fy;
    fs["cx"] >> cx;
    fs["cy"] >> cy;
    fs["bf"] >> bf;

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);

    // initialize VO object
    StereoVO stereo_vo(projMatrl,projMatrr);

    // using message_filters to get stereo callback on one topic
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, "/stereo/left/image_rect", 1);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, "/stereo/right/image_rect", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image1_sub, image2_sub);
    sync.registerCallback(boost::bind(&StereoVO::stereo_callback, &stereo_vo, _1, _2));

    std::cout << "Stereo VO Node Initialized!" << std::endl;
    
    ros::spin();
    return 0;
}
