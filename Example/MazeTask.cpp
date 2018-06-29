//
// Created by mpechac on 7. 3. 2017.
//

#include <iostream>
#include "MazeTask.h"
#include "RandomGenerator.h"

using namespace FLAB;

MazeTask::MazeTask() {
    int topology[] = {0, 0, 0, 0,
                      0, 0, 0, 2,
                      0, 0, 1, 0,
                      0, 0, 0, 0};

    maze = new Maze(topology, 4, 4, 15);
}

void MazeTask::run() {

    int action;

    for(int e = 0; e < 100; e++) {
        cout << e << endl;
        action = RandomGenerator::getInstance().random(0, 3);
        cout << maze->toString() << endl;
        maze->performAction(action);
        cout << getReward() << endl;
    }
    cout << maze->toString() << endl;
}

MazeTask::~MazeTask() {
    delete maze;
}

bool MazeTask::isFinished() {
    return (maze->actor() == maze->goal() || maze->kill());
}

double MazeTask::getReward() {
    double reward = defautPenalty;

    if (isFinished()) reward = finalReward;
    if (maze->bang()) reward = bangPenalty;
    if (maze->kill()) reward = killPenalty;

    return reward;
}
