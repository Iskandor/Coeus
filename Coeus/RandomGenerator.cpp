//
// Created by user on 14. 5. 2016.
//

#include "RandomGenerator.h"
#include <set>

RandomGenerator::RandomGenerator() {
  _mt.seed(_rd());
}

RandomGenerator::~RandomGenerator() = default;

RandomGenerator& RandomGenerator::get_instance() {
	static RandomGenerator instance;
	return instance;
}

int RandomGenerator::random(const int p_lower, const int p_upper) {
	const uniform_int_distribution<int> distribution(p_lower, p_upper);
	return distribution(_mt);
}

float RandomGenerator::random(const float p_lower, const float p_upper) {
	const uniform_real_distribution<float> distribution(p_lower, p_upper);
	return distribution(_mt);
}

float RandomGenerator::exp_random(const float p_lambda)
{
	const exponential_distribution<float> distribution(p_lambda);
	return distribution(_mt);
}

float RandomGenerator::normal_random(const float p_mean, const float p_sigma) {
	normal_distribution<float> distribution(p_mean, p_sigma);
	return distribution(_mt);
}

vector<int> RandomGenerator::choice(vector<int> *p_array, const int p_num) {
    vector<int> result;

	for(int i = 0; i < p_num; i++) {
		const auto index = static_cast<unsigned int>(random(0, p_array->size() - 1));
        result.push_back(p_array->at(index));
    }

    return vector<int>(result);
}

vector<int> RandomGenerator::choice(const int p_size, const int p_sample) {
	set<int> result;

	while(result.size() < p_sample) {
		result.insert(random(0, p_size - 1));
	}

	return vector<int>(result.begin(), result.end());
}


int RandomGenerator::choice(const float *p_prob, const int p_size) {
	int result = 0;

	const float r = random();
	float p = 0;

	for (int i = 0; i < p_size; i++) {
		if (r >= p && r < p + p_prob[i])
		{
			result = i;
			break;
		}
		p += p_prob[i];
	}

	return result;
}

void RandomGenerator::softmax(float* p_prob, const float* p_values, const int p_size, const float p_T) const
{
	float sum = 0;

	for (int i = 0; i < p_size; i++)
	{
		sum += exp(p_values[i] / p_T);
	}
	for (int i = 0; i < p_size; i++)
	{
		p_prob[i] = exp(p_values[i] / p_T) / sum;
	}

	/*
	for (int i = 0; i < p_size; i++)
	{
	if (result[i] != result[i])
	{
	break;
	}
	}
	*/
}
