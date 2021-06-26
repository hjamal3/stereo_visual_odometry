#include "vo.h"
using namespace cv;

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

/* Remove points that didn't return close enough to original position */
void removeInvalidPoints(std::vector<cv::Point2f>& pointsLeft_t0, std::vector<cv::Point2f>& pointsLeft_t1, 
    std::vector<cv::Point2f>& pointsRight_t0, FeatureSet& current_features, const std::vector<bool>& status)
{
    int index = 0;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] == false)
        {
            pointsLeft_t0.erase(pointsLeft_t0.begin() + index);
            pointsLeft_t1.erase(pointsLeft_t1.begin() + index);
            pointsRight_t0.erase(pointsRight_t0.begin() + index);
            current_features.points.erase(current_features.points.begin() + index);
            current_features.strengths.erase(current_features.strengths.begin() + index);
            current_features.ages.erase(current_features.ages.begin() + index);
        }
        else
        {
            index ++;
        }
    }

    // update current tracked points
    current_features.points = pointsLeft_t1;
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

    std::vector<cv::Point2f>  pointsLeftReturn_t0;   // feature points to check cicular matching validation

    // add new features if current number of features is below a threshold. TODO PARAM
    if (currentVOFeatures.size() < 4000)
    {
        // append new features with old features
        appendNewFeatures(imageLeft_t0, currentVOFeatures);   
        debug("[vo]: current feature set size: " + std::to_string(currentVOFeatures.points.size()));
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
    removeInvalidPoints(pointsLeft_t0, pointsLeft_t1, pointsRight_t0, currentVOFeatures, status); // can combine into one function

    debug("[vo]: number of features after circular matching: " + std::to_string(currentVOFeatures.points.size()));

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
    static cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1); // rectified undistorted images
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    static cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) << projMatrl.at<float>(0, 0), projMatrl.at<float>(0, 1), projMatrl.at<float>(0, 2),
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
    debug("[vo]: inliers size after PnP: " + std::to_string(inliers.size().height) + " out of " + std::to_string(pointsLeft_t1.size()));
    return inliers.size().height;
}