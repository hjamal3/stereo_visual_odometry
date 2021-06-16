#include "bucket.h"

Bucket::Bucket(int size){
    max_size = size;
}

int Bucket::size(){
    return features.points.size();
}

/* Score is an integer. Age is from 1-10. Strength goes up to 100. */
int Bucket::compute_score(const int age, const int strength)
{
    return age + (strength - FAST_THRESHOLD)/20;
} 

void Bucket::add_feature(const cv::Point2f point, const int age, const float strength){

    // bucket set to size 0
    if (!max_size) return;

    const int score = compute_score(age, strength);

    // won't add feature with age > 10;
    if (age < AGE_THRESHOLD)
    {
        // insert any feature before bucket is full
        if (size() < max_size)
        {
            features.points.push_back(point);
            features.ages.push_back(age);
            features.strengths.push_back(strength);
        }
        /* Bucket is full, replace feature with weakest strength. */
        else
        // weighted score of age and strength
        {            
            /* Replace weakest feature. */
            int score_min = compute_score(features.ages[0],features.strengths[0]);
            int score_min_idx = 0;
            for (int i = 1; i < size(); i++)
            {
                const int current_score = compute_score(features.ages[i],features.strengths[i]);
                if (current_score < score_min)
                {
                    score_min = current_score;
                    score_min_idx = i;
                }
            }
            if (score > score_min)
            {
                features.points[score_min_idx] = point;
                features.ages[score_min_idx] = age;
                features.strengths[score_min_idx] = strength;
            }
        }
    } 
}

Bucket::~Bucket(){
}