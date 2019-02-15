//
// Created by user on 14. 5. 2016.
//

#ifndef NEURONET_RANDOMGENERATOR_H
#define NEURONET_RANDOMGENERATOR_H


#include <random>
#include <vector>

using namespace std;

namespace FLAB {

class __declspec(dllexport) RandomGenerator {
public:
    static RandomGenerator& getInstance();
    RandomGenerator(RandomGenerator const&) = delete;
    void operator=(RandomGenerator const&)  = delete;
    ~RandomGenerator();

    float normalRandom(float p_mean = 0, float p_sigma = 1);
    int random(int p_lower, int p_upper);
    float random(float p_lower = 0, float p_upper = 1);
    vector<int> choice(vector<int>* p_array, int p_num = 1);
	vector<int> choice(int p_size, int p_sample);
    int choice(float* p_prob, int p_size);

private:
    RandomGenerator();
    random_device _rd;
    mt19937 _mt;
};

}

#endif //NEURONET_RANDOMGENERATOR_H
