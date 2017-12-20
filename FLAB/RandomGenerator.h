//
// Created by user on 14. 5. 2016.
//

#ifndef NEURONET_RANDOMGENERATOR_H
#define NEURONET_RANDOMGENERATOR_H


#include <random>
#include <vector>

using namespace std;

namespace FLAB {

class RandomGenerator {
public:
    static RandomGenerator& getInstance();
    RandomGenerator(RandomGenerator const&) = delete;
    void operator=(RandomGenerator const&)  = delete;
    ~RandomGenerator();

    double normalRandom(double p_mean = 0, double p_sigma = 1);
    int random(int p_lower, int p_upper);
    double random(double p_lower = 0, double p_upper = 1);
    vector<int> choice(vector<int>* p_array, int p_num = 1);
    int choice(double* p_prob, int p_size);

private:
    RandomGenerator();
    std::random_device _rd;
    std::mt19937 _mt;
};

}

#endif //NEURONET_RANDOMGENERATOR_H
