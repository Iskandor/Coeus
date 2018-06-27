//
// Created by mpechac on 7. 3. 2017.
//

#ifndef NEURONET_MAZE_H
#define NEURONET_MAZE_H

#include <vector>
#include <map>
#include "Environment.h"
#include "MazeAction.h"

using namespace std;

class Maze : public Environment {
public:
    Maze(int* p_topology, unsigned int p_mazeX, unsigned int p_mazeY, int p_goal);
    ~Maze();

    vector<double> getSensors();
    void performAction(double p_action);
    void reset();

    string toString();

    int actor() {
        return _actor;
    }

    int goal() {
        return _goal;
    }

    bool bang() {
        return _bang;
    }

    bool kill() {
        return _kill;
    }


private:
    vector<int> freePos();
    int moveInDir(int p_x, int p_y);

private:
    unsigned int _mazeX, _mazeY;
    vector<int> _initPos;
    vector<int> _mazeTable;
    vector<MazeAction> _actions;

    int _actor;
    int _goal;
    bool _bang;
    bool _kill;

};

#endif //NEURONET_MAZE_H
