#include "stereo_vo_node.h"
#include <stdexcept>

// quat from ekf node
void quat_callback(const::geometry_msgs::Quaternion::ConstPtr& msg)
{
	current_rot.w() = msg->w;
	current_rot.x() = msg->x;
	current_rot.y() = msg->y;
	current_rot.z() = msg->z;
	if (first_time_quat)
		vo_rot = current_rot;
}

// encoders callback
void encoders_callback(const std_msgs::Int32MultiArray::ConstPtr& msg)
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
}


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

    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;  
    matchingFeatures( imageLeft_t0, imageRight_t0,
                      imageLeft_t1, imageRight_t1, 
                      currentVOFeatures,
                      pointsLeft_t0, 
                      pointsRight_t0, 
                      pointsLeft_t1, 
                      pointsRight_t1);  

    // set new images as old images
    imageLeft_t0 = imageLeft_t1;
    imageRight_t0 = imageRight_t1;

    // display visualize feature tracks
    displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);

    // if not enough features, don't use vo
    bool use_vo = true;
    if (currentVOFeatures.size() < 10 ) //TODO should this be AND?
    {
        std::cout << "not enough features for vo" << std::endl;
		use_vo = false;        
    } else 
    {
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
	    // Integrating translations and rotations to global estimate
	    // ------------------------------------------------
	    vo_translation << translation.at<double>(0), translation.at<double>(1), translation.at<double>(2);
	    cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation); // change to axis angle	    
	    if(abs(rotation_euler[1])<0.2 && abs(rotation_euler[0])<0.2 && abs(rotation_euler[2])<0.2)
	        integrateOdometryStereo(frame_id, frame_pose, rotation, translation);
	    else
	        std::cout << "Too large rotation"  << std::endl;
	    // t_b = clock();
	    // float frame_time = 1000*(double)(t_b-t_a)/CLOCKS_PER_SEC;
	    // float fps = 1000/frame_time;
	    //cout << "[Info] frame times (ms): " << frame_time << endl;
	    //cout << "[Info] FPS: " << fps << endl;
    }

    if (use_vo)
    {
    	// rotate vo translation to rover frame
    	Eigen::Matrix<double,3,1> vo_trans_rover_frame = camera_to_world_rot._transformVector(vo_translation);

    	// rotate vo translation in rover frame to global frame
    	Eigen::Matrix<double,3,1> vo_trans_global_frame = current_rot._transformVector(vo_trans_rover_frame);

    	// add to global position
    	global_pos += vo_trans_global_frame;

    } else 
    {
    	// add to global position
    	global_pos += encoders_translation;
    }

    // update rotation for vo.
    vo_rot = current_rot; // later on do slerp between current and previous 
    vo_translation << 0,0,0;
    encoders_translation << 0,0,0;

    // publish transforms
    if (true)
    {
        static tf::TransformBroadcaster br;
		// transform of robot
        tf::Transform transform;
        //double x = global_pos[0];
        transform.setOrigin(tf::Vector3(global_pos[0],global_pos[1],global_pos[2]));
        tf::Quaternion q (current_rot.x(), current_rot.y(), current_rot.z(), current_rot.w());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "yolo"));
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

    // wheel encoders
	ros::Subscriber sub_encoders = n.subscribe("wheels", 0, encoders_callback);

    // orienation from orientation ekf
    ros::Subscriber sub_quat = n.subscribe("quat",0,quat_callback);

    std::cout << "Stereo VO Node Initialized!" << std::endl;
    
    ros::spin();
    return 0;
}
