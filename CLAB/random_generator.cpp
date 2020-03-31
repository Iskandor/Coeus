//
// Created by user on 14. 5. 2016.
//

#include "random_generator.h"
#include <set>

random_generator::random_generator() {
  _mt.seed(_rd());
}

random_generator::~random_generator() = default;

random_generator& random_generator::instance() {
	static random_generator generator;
	return generator;
}

int random_generator::random(const int p_lower, const int p_upper) {
	const uniform_int_distribution<int> distribution(p_lower, p_upper);
	return distribution(_mt);
}

float random_generator::random(const float p_lower, const float p_upper) {
	const uniform_real_distribution<float> distribution(p_lower, p_upper);
	return distribution(_mt);
}

float random_generator::exp_random(const float p_lambda)
{
	const exponential_distribution<float> distribution(p_lambda);
	return distribution(_mt);
}

float random_generator::normal_random(const float p_mean, const float p_sigma) {
	normal_distribution<float> distribution(p_mean, p_sigma);
	return distribution(_mt);
}

vector<int> random_generator::choice(vector<int> *p_array, const int p_num) {
    vector<int> result;

	for(int i = 0; i < p_num; i++) {
		const auto index = static_cast<unsigned int>(random(0, p_array->size() - 1));
        result.push_back(p_array->at(index));
    }

    return vector<int>(result);
}

vector<int> random_generator::choice(const int p_size, const int p_sample) {
	set<int> result;

	while(result.size() < p_sample) {
		result.insert(random(0, p_size - 1));
	}

	return vector<int>(result.begin(), result.end());
}


int random_generator::choice(const float *p_prob, const int p_size) {
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

void random_generator::softmax(float* p_prob, const float* p_values, const int p_size, const float p_T) const
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
}
