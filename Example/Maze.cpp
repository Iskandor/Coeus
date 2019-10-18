//
// Created by mpechac on 7. 3. 2017.
//

#include <iostream>
#include "Maze.h"
#include "RandomGenerator.h"

Maze::Maze(int *p_topology, unsigned int p_mazeX, unsigned int p_mazeY, int p_goal, bool p_stochastic) {
    for (int i = 0; i < p_mazeX * p_mazeY; i++) {
        _mazeTable.push_back(p_topology[i]);
    }

	_stochastic = p_stochastic;
    _goal = p_goal;

    _mazeX = p_mazeX;
    _mazeY = p_mazeY;

    vector<int> free = freePos();

    for (int i = 0; i < free.size(); i++) {
        if (free[i] != _goal) _initPos.push_back(free[i]);
    }

    _numActions = 4;
    _actions.push_back(MazeAction("N", 0, -1));
    _actions.push_back(MazeAction("E", 1, 0));
    _actions.push_back(MazeAction("S", 0, 1));
    _actions.push_back(MazeAction("W", -1, 0));

    reset();
}

Maze::~Maze() {

}

void Maze::reset() {
	_a = 0;
    _bang = false;
    _kill = false;
	//_actor = RandomGenerator::getInstance().choice(&_initPos)[0];
	_actor = 0;
}

vector<int> Maze::freePos() {
    vector<int> res;

    for(unsigned int i = 0; i < _mazeTable.size(); i++) {
        if (_mazeTable[i] == 0) {
            res.push_back(i);
        }
    }

    return vector<int>(res);
}

int Maze::moveInDir(int p_x, int p_y) const
{
	int pos = _actor;
	const int x = _actor % _mazeX;
	const int y = _actor / _mazeX;

	if (x + p_x >= 0 && x + p_x < _mazeX && y + p_y >= 0 && y + p_y < _mazeY)
	{
		pos = (y + p_y) * _mazeX + (x + p_x);
	}

	if (_mazeTable[pos] == 1)
	{
		pos = _actor;
	}

    return pos;
}

void Maze::performAction(float p_action) {
    //cout << _actions[p_action].Id() << endl;
	if (_stochastic) {
		if (RandomGenerator::get_instance().random() >= 0.8f && RandomGenerator::get_instance().random() < 0.9f) {
			p_action++;
			if (p_action == 4) p_action = 0;
		}
		if (RandomGenerator::get_instance().random() >= 0.9f && RandomGenerator::get_instance().random() < 1.0f) {
			p_action--;
			if (p_action < 0) p_action = 3;
		}
	}

    int newPos = moveInDir(_actions[(int)p_action].X(), _actions[(int)p_action].Y());
	_a++;

    if (newPos == _actor) {
        _bang = true;
    }
    else {
		_actor = newPos;
        if (_mazeTable[newPos] == 0) {
            _bang = false;
        }
        else if (_mazeTable[newPos] == 2) {
            _kill = true;
        }
    }
}

vector<float> Maze::getSensors() {
    vector<float> res(_mazeTable.size(), 0);

    res[_actor] = 1;

    return vector<float>(res);
}

string Maze::toString() {
    string s;

    int index;

    for(int i = 0; i < _mazeY; i++) {
        for(int j = 0; j < _mazeX; j++) {
            index = i * _mazeX + j;
            if (_actor == index) {
                s += "@";
            }
			else if (index == 0) {
				s += "S";
			}
            else if (_goal == index) {
                s += "G";
            }
            else if (_mazeTable[index] == 0) {
                s += "#";
            }
            else if (_mazeTable[index] == 1) {
                s += "#";
            }
            else if (_mazeTable[index] == 2) {
                s += "O";
            }
        }
        s += "\n";
    }

    return s;
}

string Maze::toString(int p_row)
{
	string s;

	int index;

	for (int j = 0; j < _mazeX; j++) {
		index = p_row * _mazeX + j;
		if (_actor == index) {
			s += "@";
		}
		else if (index == 0) {
			s += "S";
		}
		else if (_goal == index) {
			s += "G";
		}
		else if (_mazeTable[index] == 0) {
			s += "#";
		}
		else if (_mazeTable[index] == 1) {
			s += "#";
		}
		else if (_mazeTable[index] == 2) {
			s += "O";
		}
	}

	return s;
}
