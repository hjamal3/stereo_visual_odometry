#include "pose_estimation_node.h"
#include <stdexcept>

// quat from ekf node
void PoseEstimator::quat_callback(const::geometry_msgs::Quaternion::ConstPtr& msg)
{
	current_rot.w() = msg->w;
	current_rot.x() = msg->x;
	current_rot.y() = msg->y;
	current_rot.z() = msg->z;
	if (!orientation_init)
    {
        orientation_init = true;
		vo_rot = current_rot;
    }
}

// encoders callback
void PoseEstimator::encoders_callback(const std_msgs::Int32MultiArray::ConstPtr& msg)
{    
	int ticks_l_curr = (msg->data[0]+msg->data[2])/2.0; // total ticks left wheel
	int ticks_r_curr = (msg->data[1]+msg->data[3])/2.0; // total ticks right wheel
	
	if (first_time_enc)
	{
		ticks_l_prev = ticks_l_curr;
		ticks_r_prev = ticks_r_curr;
		first_time_enc = false;
		return;
	}

    if (!orientation_init) return; 

	// instantaneous distance moved by robot	
	double Dl = (ticks_l_curr-ticks_l_prev)/ticks_per_m;
	double Dr = (ticks_r_curr-ticks_r_prev)/ticks_per_m;
	double Dc = (Dl+Dr)/2.0;
	Eigen::Matrix<double,3,1> dpos(Dc,0,0);

	// Store previous set of readings
	ticks_l_prev = ticks_l_curr;
	ticks_r_prev = ticks_r_curr;

	// update encoders prediction
	encoders_translation += current_rot._transformVector(dpos);

    // using encoder update only
    if (!use_vo)
    {
        global_pos += encoders_translation;
        encoders_translation << 0,0,0;

        debug("[node]: Using encoder update");
        // publish transforms
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(global_pos[0],global_pos[1],global_pos[2]));
        tf::Quaternion q (current_rot.x(), current_rot.y(), current_rot.z(), current_rot.w());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "yolo"));
    }
}


PoseEstimator::PoseEstimator(cv::Mat projMatrl_, cv::Mat projMatrr_)
{
    projMatrl = projMatrl_;
    projMatrr = projMatrr_;
}

cv::Mat PoseEstimator::rosImage2CvMat(const sensor_msgs::ImageConstPtr img) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
            cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
            std::cerr << "exception" << std::endl;
            return cv::Mat();
    }
    return cv_ptr->image;
}

void PoseEstimator::stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right)
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

    frame_id++;

    // start the vo pipeline after orientation is initialized
    if (!orientation_init) return; 
    else run();
}

void PoseEstimator::run()
{
    debug("[node]: frame id " + std::to_string(frame_id));
    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;  
    matchingFeatures( imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,  currentVOFeatures,
                      pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1);  

    // visualize previous and next left image
    // displayTwoImages(imageLeft_t0, imageLeft_t1);

    // set new images as old images
    imageLeft_t0 = imageLeft_t1;
    imageRight_t0 = imageRight_t1;

    // visualize feature tracks
    //displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);

    // if not enough features don't use vo
    bool vo_usable = true;
    if (currentVOFeatures.size() < features_threshold)
    {
        debug("[node]: not enough features detected for vo: " + std::to_string(currentVOFeatures.size())  + " < " + std::to_string(features_threshold));
		vo_usable = false;        
    } else 
    {
	    // Triangulate 3D Points
	    cv::Mat points3D_t0, points4D_t0;
	    cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
	    cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

	    // PnP: computes rotation and translation between previous 3D points and next features
	    int inliers = trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t1, points3D_t0, rotation, translation);

        // PnP may not converge
        if (inliers < features_threshold)
        {
            debug("[node]: Not enough inliers from PnP: " + std::to_string(inliers) + " < " + std::to_string(features_threshold));
            vo_usable = false;
        }
        else 
        {
            Eigen::Quaternion<double> q;
            cv_rotm_to_eigen_quat(q, rotation);

            vo_translation << translation.at<double>(0), translation.at<double>(1), translation.at<double>(2);
            vo_translation = -1*(q._transformVector(vo_translation)); // pnp returns T_t1t0, invert to get T_t0t1... 

            // checking validity of VO
            double scale_translation = vo_translation.norm();
            cv::Rodrigues(rotation, rotation_rodrigues);  
            double angle = cv::norm(rotation_rodrigues, cv::NORM_L2);

            // Translation might be too big or too small, as well as rotation
            if (scale_translation < 0.01 || scale_translation > 0.1 || abs(angle) > 0.5 || abs(vo_translation(2)) > 0.04)
            {
                debug("[node]: VO rejected. Translation too small or too big or rotation too big");
                vo_usable = false;
            }
        }
    }

    if (vo_usable)
    {
    	debug("[node]: Using VO update");
    	// rotate vo translation to rover frame
    	Eigen::Matrix<double,3,1> vo_trans_rover_frame = q_bc._transformVector(vo_translation);
    	// rotate vo translation in rover frame to global frame
        Eigen::Quaternion<double> rot_ib = vo_rot.slerp(0.5, current_rot);
    	Eigen::Matrix<double,3,1> vo_trans_global_frame = rot_ib._transformVector(vo_trans_rover_frame);
    	// add to global position
    	global_pos += vo_trans_global_frame;
    } else 
    {
    	debug("[node]: Using encoder update");
    	// add to global position
    	global_pos += encoders_translation;
    }

    if (logging_path)
    {
        static auto time_init = std::chrono::steady_clock::now();
        double time_now = (double)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()- time_init).count())/1000;
        output_file << time_now << ", " << global_pos[0] << ", " << global_pos[1] << ", " << global_pos[2] << "\n";
        debug("t: " + std::to_string(time_now) + " s\n");
    }
	
	std::cout << global_pos[0] << ", " << global_pos[1] << ", " << global_pos[2] << std::endl;

    // update rotation for vo.
    vo_rot = current_rot; // later on do slerp between current and previous 
    vo_translation << 0,0,0;
    encoders_translation << 0,0,0;

    // publish transforms
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(global_pos[0],global_pos[1],global_pos[2]));
    tf::Quaternion q (current_rot.x(), current_rot.y(), current_rot.z(), current_rot.w());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "yolo"));

	nav_msgs::Odometry odom;
	odom.header.stamp = ros::Time::now();
	odom.header.frame_id = "map";
	odom.pose.pose.position.x = global_pos[0];
	odom.pose.pose.position.y = global_pos[1];
	odom.pose.pose.position.z = global_pos[2];
	odom.pose.pose.orientation.x = current_rot.x();
	odom.pose.pose.orientation.y = current_rot.y();
	odom.pose.pose.orientation.z = current_rot.z();
	odom.pose.pose.orientation.w = current_rot.w();
	(*pub_ptr).publish(odom);
}


int main(int argc, char **argv)
{

    ros::init(argc, argv, "stereo_vo_node");

    ros::NodeHandle n;

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

    // initialize pose estimator object
    PoseEstimator pose_estimator(projMatrl, projMatrr);

    // mode of operation
    n.param<bool>("use_vo", pose_estimator.use_vo, true); // accel white noise

    // log path
    n.param<bool>("logging_path", pose_estimator.logging_path, false); // accel white noise

    // using message_filters to get stereo callback on one topic
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, "/stereo/left/image_rect_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, "/stereo/right/image_rect_color", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(1), image1_sub, image2_sub);

    // use images only if vo is being used
    if (pose_estimator.use_vo)
    {
        sync.registerCallback(boost::bind(&PoseEstimator::stereo_callback, &pose_estimator, _1, _2));
    }

    // wheel encoders
	ros::Subscriber sub_encoders = n.subscribe("wheels", 0, &PoseEstimator::encoders_callback, &pose_estimator);

    // orienation from orientation ekf
    ros::Subscriber sub_quat = n.subscribe("quat", 0, &PoseEstimator::quat_callback, &pose_estimator);

	ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("odom", 1);
	pose_estimator.pub_ptr = &odom_pub;

    debug("Pose Estimator Initialized!");

    if (pose_estimator.logging_path) pose_estimator.output_file.open("/home/morphin/Desktop/output.txt");

    ros::spin();

    if (pose_estimator.logging_path) pose_estimator.output_file.close();

    return 0;
}
