//
// Created by mpechac on 7. 3. 2017.
//

#include <iostream>
#include "MazeTask.h"
#include "RandomGenerator.h"

MazeTask::MazeTask() {
    int topology[] = {0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
					  0, 0, 0, 2, 0,
					  0, 2, 0, 0, 0,
					  0, 0, 0, 0, 0};

    maze = new Maze(topology, 5, 5, 24);
}

void MazeTask::run() {

    int action;

    for(int e = 0; e < 100; e++) {
        cout << e << endl;
        action = RandomGenerator::get_instance().random(0, 3);
        cout << maze->toString() << endl;
        maze->performAction(action);
        cout << getReward() << endl;
    }
    cout << maze->toString() << endl;
}

MazeTask::~MazeTask() {
    delete maze;
}

bool MazeTask::isFinished() const {
    return (isWinner() || maze->kill() || maze->moves() > 100);
}

bool MazeTask::isWinner() const
{
	return maze->actor() == maze->goal();
}

float MazeTask::getReward() const {
    float reward = defautPenalty;

    if (isWinner()) reward = finalReward;
    if (maze->bang()) reward = bangPenalty;
    if (maze->kill()) reward = killPenalty;

    return reward;
}
