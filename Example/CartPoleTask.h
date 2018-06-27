//
// Created by mpechac on 21. 3. 2017.
//

#ifndef NEURONET_CARTPOLETASK_H
#define NEURONET_CARTPOLETASK_H


#include "CartPole.h"


class CartPoleTask {
public:
    CartPoleTask(int p_maxT = 1000);
    ~CartPoleTask();

    void run();
    void reset();

    bool isFinished();
    bool failed();
    double getReward();

    CartPole *getEnvironment() {
        return _cartPole;
    }

    int getT() {
        return _t;
    }

private:
    const double defautPenalty = -1;
    const double bangPenalty = -1;
    const double killPenalty = -10;
    const double finalReward = 10;

    CartPole *_cartPole;

    int _t;
    int _maxT;
};


#endif //NEURONET_CARTPOLETASK_H
