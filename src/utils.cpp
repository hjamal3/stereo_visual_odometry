#include "stereo_visual_odometry/utils.h"
#include "nav_msgs/Odometry.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

// --------------------------------
// Visualization
// --------------------------------
void drawFeaturePoints(cv::Mat image, std::vector<cv::Point2f>& points)
{
    int radius = 2;
    
    for (int i = 0; i < points.size(); i++)
    {
        circle(image, cv::Point(points[i].x, points[i].y), radius, CV_RGB(255,255,255));
    }
}

void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, std::vector<Matrix>& pose_matrix_gt, float fps, bool show_gt)
{
    // draw estimated trajectory 
    int x = int(pose.at<double>(0)) + 300;
    int y = int(pose.at<double>(2)) + 100;
    circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);

    if (show_gt)
    {
      // draw ground truth trajectory 
      cv::Mat pose_gt = cv::Mat::zeros(1, 3, CV_64F);
      
      pose_gt.at<double>(0) = pose_matrix_gt[frame_id].val[0][3];
      pose_gt.at<double>(1) = pose_matrix_gt[frame_id].val[0][7];
      pose_gt.at<double>(2) = pose_matrix_gt[frame_id].val[0][11];
      x = int(pose_gt.at<double>(0)) + 300;
      y = int(pose_gt.at<double>(2)) + 100;
      circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,255,0), 2);
    }
    // print info

    // rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    // sprintf(text, "FPS: %02f", fps);
    // putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    cv::imshow( "Trajectory", trajectory );
    cv::waitKey(1);
}



// --------------------------------
// Transformation
// --------------------------------


void integrateOdometryStereo(int frame_i, cv::Mat& frame_pose, const cv::Mat& rotation, const cv::Mat& translation_stereo, ros::Publisher p)
{

    // std::cout << "rotation" << rotation << std::endl;
    // std::cout << "translation_stereo" << translation_stereo << std::endl;

    cv::Mat rigid_body_transformation;

    cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

    cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
    cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

    // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

    double scale = sqrt((translation_stereo.at<double>(0))*(translation_stereo.at<double>(0)) 
                        + (translation_stereo.at<double>(1))*(translation_stereo.at<double>(1))
                        + (translation_stereo.at<double>(2))*(translation_stereo.at<double>(2))) ;

    // frame_pose = frame_pose * rigid_body_transformation;
    std::cout << "scale: " << scale << std::endl;

    rigid_body_transformation = rigid_body_transformation.inv();
    // if ((scale>0.1)&&(translation_stereo.at<double>(2) > translation_stereo.at<double>(0)) && (translation_stereo.at<double>(2) > translation_stereo.at<double>(1))) 
    ///if (scale > 0.05 && scale < 10) 
    if (scale > 0.001 && scale < 10) // WHY DO WE NEED THIS
    {
      // std::cout << "Rpose" << Rpose << std::endl;

      frame_pose = frame_pose * rigid_body_transformation;


      /*for display purposes TODO add covariance TODO remove*/
      nav_msgs::Odometry msg;
      msg.header.frame_id = "map";
      ros::Time cur = ros::Time::now();
      msg.header.stamp.sec = cur.toSec();
      msg.header.stamp.nsec = cur.toNSec();

      Eigen::Matrix3d r = Eigen::Matrix<double,3,3>::Zero();
      
      // copy rotation matrix from frame_pose to r
      for(int i=0; i<3; i++)
	      for(int j=0; j<3; j++)
		      r(i,j) = frame_pose.at<double>(i,j);

      Eigen::Quaterniond q(r);

      //TODO change to pointer accesses
      msg.pose.pose.position.x = frame_pose.at<double>(0,3);
      msg.pose.pose.position.y = frame_pose.at<double>(1,3);
      msg.pose.pose.position.z = frame_pose.at<double>(2,3);
      msg.pose.pose.orientation.x = q.x();
      msg.pose.pose.orientation.y = q.y();
      msg.pose.pose.orientation.z = q.z();
      msg.pose.pose.orientation.w = q.w();

      p.publish(msg);

      std::cout << "frame_pose" << frame_pose << std::endl;

    }
    else 
    {
     std::cout << "[WARNING] scale below 0.1, or incorrect translation" << std::endl;
    }
}

bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
     
    return  norm(I, shouldBeIdentity) < 1e-6;
     
}
 
// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{
 
    assert(isRotationMatrix(R));
     
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
     
}

// --------------------------------
// I/O
// --------------------------------

void loadImageLeft(cv::Mat& image_color, cv::Mat& image_gary, int frame_id, std::string filepath){
    char file[200];
    sprintf(file, "image_0/%06d.png", frame_id);
    std::string filename = filepath + std::string(file);
    image_color = cv::imread(filename, cv::IMREAD_COLOR);
    cvtColor(image_color, image_gary, cv::COLOR_BGR2GRAY);
}

void loadImageRight(cv::Mat& image_color, cv::Mat& image_gary, int frame_id, std::string filepath){
    char file[200];
    sprintf(file, "image_1/%06d.png", frame_id);
    std::string filename = filepath + std::string(file);
    image_color = cv::imread(filename, cv::IMREAD_COLOR);
    cvtColor(image_color, image_gary, cv::COLOR_BGR2GRAY);
}











