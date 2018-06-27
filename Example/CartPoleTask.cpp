//
// Created by mpechac on 21. 3. 2017.
//

#include <math.h>
#include <iostream>
#include "CartPoleTask.h"
#include "RandomGenerator.h"

using namespace FLAB;

CartPoleTask::CartPoleTask(int p_maxT) {
    _maxT = p_maxT;
    _cartPole = new CartPole();
    reset();
}

CartPoleTask::~CartPoleTask() {
    delete _cartPole;
}

void CartPoleTask::run() {
    double action;

    for(int e = 0; e < 1000; e++) {
        cout << e << endl;
        action = RandomGenerator::getInstance().random(-1, 1);
        cout << _cartPole->toString() << endl;
        _cartPole->performAction(action);
        cout << getReward() << endl;

        if (isFinished()) {
            reset();
        }
    }
    cout << _cartPole->toString() << endl;
}

bool CartPoleTask::isFinished() {
    bool res = false;

    if (fabs(_cartPole->getPoleAngle()) > 0.7) {
        res = true;
    }
    if (fabs(_cartPole->getCartPosition()) > 2.4) {
        res = true;
    }
    if (_t >= _maxT) {
        res = true;
    }

    return res;
}

double CartPoleTask::getReward() {
    double reward = 0;

    _t++;
    if (_cartPole->getPoleAngle() < 0.05 && fabs(_cartPole->getCartPosition()) < 0.05) {
        reward = 0;
    }
    else if (_cartPole->getPoleAngle() > 0.7 || fabs(_cartPole->getCartPosition()) > 2.4) {
        reward = -2 * (_maxT - _t);
    }
    else {
        reward = -1;
    }

    return reward;
}

void CartPoleTask::reset() {
    _t = 0;
    _cartPole->reset();
}

bool CartPoleTask::failed() {
    return (_t < _maxT);
}
