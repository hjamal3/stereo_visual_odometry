#include "stereo_visual_odometry/visualOdometry.h"
using namespace cv;

cv::Mat euler2rot(cv::Mat& rotationMatrix, const cv::Mat & euler)
{

    double x = euler.at<double>(0);
    double y = euler.at<double>(1);
    double z = euler.at<double>(2);

    // Assuming the angles are in radians.
    double ch = cos(z);
    double sh = sin(z);
    double ca = cos(y);
    double sa = sin(y);
    double cb = cos(x);
    double sb = sin(x);

    double m00, m01, m02, m10, m11, m12, m20, m21, m22;

    m00 = ch * ca;
    m01 = sh*sb - ch*sa*cb;
    m02 = ch*sa*sb + sh*cb;
    m10 = sa;
    m11 = ca*cb;
    m12 = -ca*sb;
    m20 = -sh*ca;
    m21 = sh*sa*cb + ch*sb;
    m22 = -sh*sa*sb + ch*cb;

    rotationMatrix.at<double>(0,0) = m00;
    rotationMatrix.at<double>(0,1) = m01;
    rotationMatrix.at<double>(0,2) = m02;
    rotationMatrix.at<double>(1,0) = m10;
    rotationMatrix.at<double>(1,1) = m11;
    rotationMatrix.at<double>(1,2) = m12;
    rotationMatrix.at<double>(2,0) = m20;
    rotationMatrix.at<double>(2,1) = m21;
    rotationMatrix.at<double>(2,2) = m22;

    return rotationMatrix;
}

void checkValidMatch(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_return, std::vector<bool>& status, int threshold)
{
    int offset;
    for (int i = 0; i < points.size(); i++)
    {
        offset = std::max(std::abs(points[i].x - points_return[i].x), std::abs(points[i].y - points_return[i].y));
        // std::cout << offset << ", ";

        if(offset > threshold)
        {
            status.push_back(false);
        }
        else
        {
            status.push_back(true);
        }
    }
}

void removeInvalidPoints(std::vector<cv::Point2f>& points, const std::vector<bool>& status)
{
    int index = 0;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] == false)
        {
            points.erase(points.begin() + index);
        }
        else
        {
            index ++;
        }
    }
}

void matchingFeatures(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<cv::Point2f>&  pointsLeft_t0, 
                      std::vector<cv::Point2f>&  pointsRight_t0, 
                      std::vector<cv::Point2f>&  pointsLeft_t1, 
                      std::vector<cv::Point2f>&  pointsRight_t1)
{
    // ----------------------------
    // Feature detection using FAST
    // ----------------------------
    std::vector<cv::Point2f>  pointsLeftReturn_t0;   // feature points to check cicular mathcing validation

    // add new features if current number of features is below a threshold. TODO PARAM
    if (currentVOFeatures.size() < 2000)
    {

        // append new features with old features
        appendNewFeatures(imageLeft_t0, currentVOFeatures);   
        //std::cout << "Current feature set size: " << currentVOFeatures.points.size() << std::endl;
    }

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching
    // --------------------------------------------------------
    int bucket_size = std::min(imageLeft_t0.rows,imageLeft_t0.cols)/10; // TODO PARAM
    int features_per_bucket = 1; // TODO PARAM
    std::cout << "number of features before bucketing: " << currentVOFeatures.points.size() << std::endl;

    // feature detector points before bucketing
    //displayPoints(imageLeft_t0,currentVOFeatures.points);

    // filter features in currentVOFeatures so that one per bucket
    bucketingFeatures(imageLeft_t0, currentVOFeatures, bucket_size, features_per_bucket);
    pointsLeft_t0 = currentVOFeatures.points;

    // feature detector points after bucketing
    //displayPoints(imageLeft_t0,currentVOFeatures.points);

    #if USE_CUDA
        circularMatching_gpu(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);
    #else
	    circularMatching(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);
    #endif

    // check if circled back points are in range of original points
    std::vector<bool> status;
    checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 0);
    removeInvalidPoints(pointsLeft_t0, status); // can combine into one function
    removeInvalidPoints(pointsLeft_t1, status);
    removeInvalidPoints(pointsRight_t0, status);
    removeInvalidPoints(pointsRight_t1, status);

    std::cout << "number of features after bucketing: " << currentVOFeatures.points.size() << std::endl;

    // update current tracked points
    currentVOFeatures.points = pointsLeft_t1;

    std::cout << "number of features after circular matching: " << currentVOFeatures.points.size() << std::endl;

    // feature detector points after circular matching
    //displayPoints(imageLeft_t0,currentVOFeatures.points);

}

// The transformation is parameterized using 6 parameters: 3 for rotation, 3 for translation
struct Error2D {
  Error2D(double pt_x, double pt_y, double pt_z, double pix_x, double pix_y)
      : pt_x(pt_x), pt_y(pt_y), pt_z(pt_z), pix_x(pix_x), pix_y(pix_y) {}
  template <typename T>
  bool operator()(const T* const camera,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    T point[3];
    point[0] = T(pt_x);
    point[1] = T(pt_y);
    point[2] = T(pt_z);
    ceres::AngleAxisRotatePoint(camera, point, p);
  
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // z transform
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Compute final projected point position.
    const T fx = T(718.8560);
    const T fy = T(718.8560);
    const T cx = T(607.1928);
    const T cy = T(185.2157);
    T predicted_x = fx * xp + cx;
    T predicted_y = fy * yp + cy;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - pix_x;
    residuals[1] = predicted_y - pix_y;

    return true;
  }
  const double pt_x;
  const double pt_y;
  const double pt_z;
  const double pix_x;
  const double pix_y;
};



// optimizes rotation and translation with nonlinear optimization (minimizing reprojection error).
// optimizes inliers from PnP only.
void optimize_transformation(cv::Mat& rotation, cv::Mat& translation, cv::Mat & points3D, 
    std::vector<cv::Point2f>& pointsLeft, cv::Mat& inliers, cv::Mat & projection_matrix)
{
    static bool init = false;
    if (!init)
    {
        init = true;
        google::InitGoogleLogging("vo"); // TODO PUT SOMEWHERE ELSE
    }
    // initial transformation
    double* camera = new double[6];
    camera[0] = rotation.at<double>(0,0);
    camera[1] = rotation.at<double>(1,0);
    camera[2] = rotation.at<double>(2,0);
    camera[3] = translation.at<double>(0,0);
    camera[4] = translation.at<double>(1,0);
    camera[5] = translation.at<double>(2,0);

    int num_inliers = int(inliers.size().height);
    double camera_init[6];
    for (int i = 0; i < 6; i++) camera_init[i] = camera[i];
    ceres::Problem problem;
    for (int i = 0; i < num_inliers; i++)
    {
        int idx = inliers.at<int>(0,i);
        problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Error2D, 2, 6>( // dimension of residual, dimension of camera
            new Error2D(points3D.at<float>(idx,0), points3D.at<float>(idx,1), points3D.at<float>(idx,2), pointsLeft[idx].x, pointsLeft[idx].y)),
        NULL,
        camera);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Initial r: " << camera_init[0] << " " << camera_init[1] << " " << camera_init[2] << "\n";
    std::cout << "Initial t: " << camera_init[3] << " " << camera_init[4] << " " << camera_init[5] << "\n";
    std::cout << "Final r: " << camera[0] << " " << camera[1] << " " << camera[2] << "\n";
    std::cout << "Final t: " << camera[3] << " " << camera[4] << " " << camera[5] << "\n";

}


void trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::Point2f>&  pointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation,
                         bool mono_rotation)
{

    // Calculate frame to frame transformation
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);   
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) << projMatrl.at<float>(0, 0), projMatrl.at<float>(0, 1), projMatrl.at<float>(0, 2),
                                                projMatrl.at<float>(1, 0), projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2),
                                                projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2), projMatrl.at<float>(1, 3));

    int iterationsCount = 500;        // number of Ransac iterations.
    float reprojectionError = .5;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.999;          // RANSAC successful confidence.
    bool useExtrinsicGuess = true;
    int flags =cv::SOLVEPNP_ITERATIVE;

    #if 1
    cv::Mat inliers; 
    cv::solvePnPRansac( points3D_t0, pointsLeft_t1, intrinsic_matrix, distCoeffs, rvec, translation,
                        useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                        inliers, flags );
    #endif
    #if 0
    std::vector<int> inliers;
    cv::cuda::solvePnPRansac(points3D_t0.t(), cv::Mat(1, (int)pointsLeft_t1.size(), CV_32FC2, &pointsLeft_t1[0]),
                        intrinsic_matrix, cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)),
                        rvec, translation, false, 200, 0.5, 20, &inliers);
    #endif

    // nonlinear optimization after, minimizing reprojection error
    //optimize_transformation(rvec,translation,points3D_t0,pointsLeft_t1,inliers, projMatrl);
    cv::Rodrigues(rvec, rotation);
    std::cout << "[trackingFrame2Frame] inliers size: " << inliers.size()  << " out of " << pointsLeft_t1.size() << std::endl;

}

void displayPoints(cv::Mat& image, std::vector<cv::Point2f>&  points)
{
    int radius = 2;
    cv::Mat vis;

    cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR, 3);

    for (int i = 0; i < points.size(); i++)
    {
        cv::circle(vis, cv::Point(points[i].x, points[i].y), radius, CV_RGB(0,255,0));
    }

    cv::imshow("vis ", vis );  
    cv::waitKey(1);
}

void displayTracking(cv::Mat& imageLeft_t1, 
                     std::vector<cv::Point2f>&  pointsLeft_t0,
                     std::vector<cv::Point2f>&  pointsLeft_t1)
{
    // -----------------------------------------
    // Display feature racking
    // -----------------------------------------
    int radius = 2;
    cv::Mat vis;

    cv::cvtColor(imageLeft_t1, vis, cv::COLOR_GRAY2BGR, 3);

    for (int i = 0; i < pointsLeft_t0.size(); i++)
    {
      cv::circle(vis, cv::Point(pointsLeft_t0[i].x, pointsLeft_t0[i].y), radius, CV_RGB(0,255,0));
    }

    for (int i = 0; i < pointsLeft_t1.size(); i++)
    {
      cv::circle(vis, cv::Point(pointsLeft_t1[i].x, pointsLeft_t1[i].y), radius, CV_RGB(255,0,0));
    }

    for (int i = 0; i < pointsLeft_t1.size(); i++)
    {
      cv::line(vis, pointsLeft_t0[i], pointsLeft_t1[i], CV_RGB(0,255,0));
    }

    cv::imshow("vis ", vis );  
    cv::waitKey(1);
}
