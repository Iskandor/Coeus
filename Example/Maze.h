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

    vector<float> getSensors() override;
    void performAction(float p_action) override;
    void reset() override;

    string toString();

    int actor() const {
        return _actor;
    }

    int goal() const {
        return _goal;
    }

    bool bang() const {
        return _bang;
    }

    bool kill() const {
        return _kill;
    }

	int moves() const {
		return _a;
    }

	unsigned int mazeX() const
	{
		return _mazeX;
    }

	unsigned int mazeY() const
	{
		return _mazeY;
	}

private:
    vector<int> freePos();
    int moveInDir(int p_x, int p_y) const;

    unsigned int _mazeX, _mazeY;
    vector<int> _initPos;
    vector<int> _mazeTable;
    vector<MazeAction> _actions;

    int _actor;
    int _goal;
    bool _bang;
    bool _kill;

	int _a;

};

#endif //NEURONET_MAZE_H
