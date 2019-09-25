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
    float getReward() const;

    Maze *getEnvironment() {
        return maze;
    }


private:
    const float defautPenalty = 0;
    const float bangPenalty = 0;
    const float killPenalty = 0;
    const float finalReward = 1;

    Maze *maze;
};


#endif //NEURONET_MAZETASK_H
