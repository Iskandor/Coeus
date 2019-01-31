//
// Created by mpechac on 7. 3. 2017.
//

#ifndef NEURONET_MAZETASK_H
#define NEURONET_MAZETASK_H


#include "Maze.h"

class MazeTask {
public:
    MazeTask();
    ~MazeTask();

    void run();

    bool isFinished() const;
	bool isWinner() const;
    double getReward() const;

    Maze *getEnvironment() {
        return maze;
    }


private:
    const double defautPenalty = 0;
    const double bangPenalty = -1;
    const double killPenalty = -10;
    const double finalReward = 100;

    Maze *maze;
};


#endif //NEURONET_MAZETASK_H
