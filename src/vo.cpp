#include "vo.h"
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

/* Removes any feature points that did not circle back to their original location, with threshold.*/
void checkValidMatch(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_return, std::vector<bool>& status, int threshold)
{
    int offset;
    for (int i = 0; i < points.size(); i++)
    {
        offset = std::max(std::abs(points[i].x - points_return[i].x), std::abs(points[i].y - points_return[i].y));
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
    // Feature detection using FAST and bucketing
    // ----------------------------
    currentVOFeatures.points.clear();
    currentVOFeatures.ages.clear();
    currentVOFeatures.strengths.clear();

    std::vector<cv::Point2f>  pointsLeftReturn_t0;   // feature points to check cicular mathcing validation

    // add new features if current number of features is below a threshold. TODO PARAM
    if (currentVOFeatures.size() < 2000)
    {
        // append new features with old features
        appendNewFeatures(imageLeft_t0, currentVOFeatures);   
        std::cout << "Current feature set size: " << currentVOFeatures.points.size() << std::endl;
    }
    // left image points are the tracked features
    pointsLeft_t0 = currentVOFeatures.points;

    // --------------------------------------------------------
    // Feature tracking using KLT tracker and circular matching
    // --------------------------------------------------------
    if (currentVOFeatures.points.size() == 0) return; // early exit

    // Debugging: show two images
    // displayTwoImages(imageLeft_t0, imageLeft_t1);
    // std::cout << imageLeft_t1-imageLeft_t0 << std::endl;

    #if USE_CUDA
        circularMatching_gpu(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);
    #else
	    circularMatching(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);
    #endif

    // check if circled back points are in range of original points
    std::vector<bool> status;
    checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 1);
    removeInvalidPoints(pointsLeft_t0, status); // can combine into one function
    removeInvalidPoints(pointsLeft_t1, status);
    removeInvalidPoints(pointsRight_t0, status);
    removeInvalidPoints(pointsRight_t1, status);

    // update current tracked points
    currentVOFeatures.points = pointsLeft_t1;

    std::cout << "number of features after circular matching: " << currentVOFeatures.points.size() << std::endl;

    // feature detector points after circular matching
    //displayPoints(imageLeft_t0,currentVOFeatures.points);

}


int trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::Point2f>&  pointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation)
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

    cv::Rodrigues(rvec, rotation);
    std::cout << "[trackingFrame2Frame] inliers size: " << inliers.size()  << " out of " << pointsLeft_t1.size() << std::endl;
    return inliers.size().height;
}
