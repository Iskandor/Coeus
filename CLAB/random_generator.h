//
// Created by user on 14. 5. 2016.
//
#pragma once

#include <random>
#include <vector>

using namespace std;

class __declspec(dllexport) random_generator {
public:
    static random_generator& instance();
    random_generator(random_generator const&) = delete;
    void operator=(random_generator const&)  = delete;
    ~random_generator();

	float exp_random(float p_lambda = 1);
    float normal_random(float p_mean = 0, float p_sigma = 1);
    int random(int p_lower, int p_upper);
    float random(float p_lower = 0, float p_upper = 1);
    vector<int> choice(vector<int>* p_array, int p_num = 1);
	vector<int> choice(int p_size, int p_sample);
    int choice(const float* p_prob, int p_size);
	void softmax(float* p_prob, const float* p_values, int p_size, float p_T) const;

private:
    random_generator();
    random_device _rd;
    mt19937 _mt;
};
