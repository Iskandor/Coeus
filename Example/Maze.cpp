//
// Created by mpechac on 7. 3. 2017.
//

#include <iostream>
#include "Maze.h"
#include "RandomGenerator.h"

Maze::Maze(int *p_topology, unsigned int p_mazeX, unsigned int p_mazeY, int p_goal, bool p_stochastic) : IEnvironment() {
	init(p_topology, p_mazeX, p_mazeY, p_goal, p_stochastic);
}

Maze::Maze()
= default;

Maze::Maze(Maze& p_copy) : IEnvironment(p_copy)
{
	init(p_copy._topology, p_copy._mazeX, p_copy._mazeY, p_copy._goal, p_copy._stochastic);
}

Maze::~Maze() = default;

Maze& Maze::operator=(const Maze& p_copy)
{
	init(p_copy._topology, p_copy._mazeX, p_copy._mazeY, p_copy._goal, p_copy._stochastic);
	return *this;
}

void Maze::init(int* p_topology, unsigned p_mazeX, unsigned p_mazeY, int p_goal, bool p_stochastic)
{
	delete[] _topology;
	_topology = new int[p_mazeY * p_mazeX];
	memcpy(_topology, p_topology, sizeof(int) * p_mazeY * p_mazeX);
	_state_dim = p_mazeY * p_mazeX;
	_action_dim = 4;

	for (int i = 0; i < p_mazeX * p_mazeY; i++) {
		_maze_table.push_back(p_topology[i]);
	}

	_stochastic = p_stochastic;
	_goal = p_goal;

	_mazeX = p_mazeX;
	_mazeY = p_mazeY;

	vector<int> free = freePos();

	for (int i = 0; i < free.size(); i++) {
		if (free[i] != _goal) _init_pos.push_back(free[i]);
	}

	_actions.emplace_back("N", 0, -1);
	_actions.emplace_back("E", 1, 0);
	_actions.emplace_back("S", 0, 1);
	_actions.emplace_back("W", -1, 0);

	Maze::reset();
}

Tensor Maze::get_state()
{
	Tensor res({ _state_dim }, Tensor::ZERO);

	res[_actor] = 1;

	return res;
}

void Maze::do_action(Tensor& p_action)
{
	int action = p_action.max_value_index();
	//cout << _actions[p_action].Id() << endl;
	if (_stochastic) {
		if (RandomGenerator::get_instance().random() >= 0.8f && RandomGenerator::get_instance().random() < 0.9f) {
			action++;
			if (action == 4) action = 0;
		}
		if (RandomGenerator::get_instance().random() >= 0.9f && RandomGenerator::get_instance().random() < 1.0f) {
			action--;
			if (action < 0) action = 3;
		}
	}

	const int new_pos = moveInDir(_actions[action].X(), _actions[action].Y());
	_a++;

	if (new_pos == _actor) {
		_bang = true;
	}
	else {
		_actor = new_pos;
		if (_maze_table[new_pos] == 0) {
			_bang = false;
		}
		else if (_maze_table[new_pos] == 2) {
			_kill = true;
		}
	}
}

float Maze::get_reward()
{
	float reward = defautPenalty;

	if (is_winner()) reward = finalReward;
	if (_bang) reward = bangPenalty;
	if (_kill) reward = killPenalty;

	return reward;
}

bool Maze::is_finished()
{
	return (is_winner() || _kill || moves() > 100);
}

void Maze::reset() {
	_a = 0;
    _bang = false;
    _kill = false;
	//_actor = RandomGenerator::getInstance().choice(&_init_pos)[0];
	_actor = 0;
}

bool Maze::is_winner() const
{
	return _actor == _goal;
}

vector<int> Maze::freePos() {
    vector<int> res;

    for(unsigned int i = 0; i < _maze_table.size(); i++) {
        if (_maze_table[i] == 0) {
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

	if (_maze_table[pos] == 1)
	{
		pos = _actor;
	}

    return pos;
}

string Maze::toString() {
    string s;

    for(int i = 0; i < _mazeY; i++) {
        for(int j = 0; j < _mazeX; j++) {
	        const int index = i * _mazeX + j;
            if (_actor == index) {
                s += "@";
            }
			else if (index == 0) {
				s += "S";
			}
            else if (_goal == index) {
                s += "G";
            }
            else if (_maze_table[index] == 0) {
                s += "#";
            }
            else if (_maze_table[index] == 1) {
                s += "#";
            }
            else if (_maze_table[index] == 2) {
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

	for (int j = 0; j < _mazeX; j++) {
		const int index = p_row * _mazeX + j;
		if (_actor == index) {
			s += "@";
		}
		else if (index == 0) {
			s += "S";
		}
		else if (_goal == index) {
			s += "G";
		}
		else if (_maze_table[index] == 0) {
			s += "#";
		}
		else if (_maze_table[index] == 1) {
			s += "#";
		}
		else if (_maze_table[index] == 2) {
			s += "O";
		}
	}

	return s;
}
