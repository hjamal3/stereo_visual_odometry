#ifndef BUCKET_H
#define BUCKET_H

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

class Bucket
{

public:
    int id;
    int max_size;

    FeatureSet features;

    Bucket(int);
    ~Bucket();

    void add_feature(const cv::Point2f, const int, const float);
    void get_features(FeatureSet&);
    int compute_score(const int age, const int strength);

    int size();
    
};

#endif
