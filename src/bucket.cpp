#include "bucket.h"

Bucket::Bucket(int size){
    max_size = size;
}

int Bucket::size(){
    return features.points.size();
}


void Bucket::add_feature(const cv::Point2f point, const int age, const float strength){

    // bucket set to size 0
    if (!max_size) return;

    // won't add feature with age > 10;
    const int age_threshold = 10;
    if (age < age_threshold)
    {
        // insert any feature before bucket is full
        if (size() < max_size)
        {
            features.points.push_back(point);
            features.ages.push_back(age);
            features.strengths.push_back(strength);
        }
        else
        // OLD: insert feature with old age and remove youngest one
        // TODO: weighted sum of age and strength
        {
            /* Replace weakest feature. */
            float strength_min = features.strengths[0];
            int strength_min_idx = 0;
            for (int i = 0; i < size(); i++)
            {
                const float current_strength = features.strengths[i];
                if (current_strength < strength_min)
                {
                    strength_min = current_strength;
                    strength_min_idx = i;
                }
            }
            // if (strength > strength_min)
            // {
            features.points[strength_min_idx] = point;
            features.ages[strength_min_idx] = age;
            features.strengths[strength_min_idx] = strength;
            // }

            // int age_min = features.ages[0];
            // int age_min_idx = 0;
            // for (int i = 0; i < size(); i++)
            // {
            //     age_curr = features.ages[i];
            //     if (age < age_min)
            //     {
            //         age_min = age;
            //         age_min_idx = i;
            //     }
            // }
        }
    } 

}

// void Bucket::get_features(FeatureSet& current_features){

//     current_features.points.insert(current_features.points.end(), features.points.begin(), features.points.end());
//     current_features.ages.insert(current_features.ages.end(), features.ages.begin(), features.ages.end());
// }

Bucket::~Bucket(){
}