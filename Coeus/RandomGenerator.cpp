//
// Created by user on 14. 5. 2016.
//

#include "RandomGenerator.h"

using namespace FLAB;

RandomGenerator::RandomGenerator() {
  _mt.seed(_rd());
}

RandomGenerator::~RandomGenerator() {
}

RandomGenerator& RandomGenerator::getInstance() {
  static RandomGenerator instance;
  return instance;
}

int RandomGenerator::random(int p_lower, int p_upper) {
  uniform_int_distribution<int> distribution(p_lower, p_upper);
  return distribution(_mt);
}

float RandomGenerator::random(float p_lower, float p_upper) {
  uniform_real_distribution<float> distribution(p_lower, p_upper);
  return distribution(_mt);
}

float RandomGenerator::normalRandom(float p_mean, float p_sigma) {
  normal_distribution<float> distribution(p_mean, p_sigma);
  return distribution(_mt);
}

vector<int> RandomGenerator::choice(vector<int> *p_array, int p_num) {
    vector<int> result;
    unsigned int index;

    for(int i = 0; i < p_num; i++) {
        index = (unsigned int) random(0, p_array->size()-1);
        result.push_back(p_array->at(index));
    }

    return vector<int>(result);
}

vector<int> RandomGenerator::choice(const int p_size, const int p_sample) {
	vector<int> result;

	for (int i = 0; i < p_sample; i++) {
		result.push_back(random(0, p_size - 1));
	}

	return vector<int>(result);
}


int RandomGenerator::choice(float *p_prob, int p_size) {

    vector<int> candidates;

    while(candidates.empty()) {
        for(int i = 0; i < p_size; i++) {

            if (random() < p_prob[i]) {
                candidates.push_back(i);
            }
        }
    }

    int result = candidates[0];

    for(int i = 0; i < candidates.size(); i++) {
        if (p_prob[result] < p_prob[candidates[i]]) {
            result = candidates[i];
        }
    }

    return result;
}
