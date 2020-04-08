//
// Created by mpechac on 7. 3. 2017.
//

#include <iostream>
#include "maze.h"
#include "random_generator.h"

maze::maze(int *p_topology, unsigned int p_mazeX, unsigned int p_mazeY, int p_goal, bool p_stochastic)
{
	_state_dim = p_mazeY * p_mazeX;
	_action_dim = 4;
	_topology = nullptr;
	init(p_topology, p_mazeX, p_mazeY, p_goal, p_stochastic);
}

maze::maze()
{
	_topology = nullptr;
}

maze::maze(maze& p_copy) : ienvironment(p_copy)
{
	init(p_copy._topology, p_copy._mazeX, p_copy._mazeY, p_copy._goal, p_copy._stochastic);
}

maze::~maze()
{
	delete[] _topology;
}

maze& maze::operator=(const maze& p_copy)
{
	init(p_copy._topology, p_copy._mazeX, p_copy._mazeY, p_copy._goal, p_copy._stochastic);
	return *this;
}

void maze::init(int* p_topology, unsigned p_mazeX, unsigned p_mazeY, int p_goal, bool p_stochastic)
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

	maze::reset();
}

tensor maze::get_state()
{
	tensor res({ _state_dim }, tensor::ZERO);

	res[_actor] = 1;

	return res;
}

void maze::set_state(tensor& p_state)
{
	_actor = p_state.max_index()[0];
}

void maze::do_action(tensor& p_action)
{
	int action = p_action.max_index()[0];
	if (_stochastic) {
		if (random_generator::instance().random() >= 0.8f && random_generator::instance().random() < 0.9f) {
			action++;
			if (action == 4) action = 0;
		}
		if (random_generator::instance().random() >= 0.9f && random_generator::instance().random() < 1.0f) {
			action--;
			if (action < 0) action = 3;
		}
	}

	int action_x;
	int action_y;

	switch(action)
	{
	case 0:
		action_x = 0;
		action_y = -1;
		break;
	case 1:
		action_x = 1;
		action_y = 0;
		break;
	case 2:
		action_x = 0;
		action_y = 1;
		break;
	case 3:
		action_x = -1;
		action_y = 0;
		break;
	default:
		printf("Wrong action\n");
		action_x = 0;
		action_y = 0;
	}

	const int new_pos = move(action_x, action_y);
	_steps++;

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

float maze::get_reward()
{
	float reward = defautPenalty;

	if (is_winner()) reward = finalReward;
	if (_bang) reward = bangPenalty;
	if (_kill) reward = killPenalty;

	return reward;
}

bool maze::is_finished()
{
	return (is_winner() || _kill || moves() > 100);
}

void maze::reset() {
	_steps = 0;
    _bang = false;
    _kill = false;
	//_actor = RandomGenerator::getInstance().choice(&_init_pos)[0];
	_actor = 0;
}

bool maze::is_winner() const
{
	return _actor == _goal;
}

vector<int> maze::freePos() {
    vector<int> res;

    for(unsigned int i = 0; i < _maze_table.size(); i++) {
        if (_maze_table[i] == 0) {
            res.push_back(i);
        }
    }

    return vector<int>(res);
}

int maze::move(int p_x, int p_y) const
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

string maze::to_string() {
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

string maze::to_string(int p_row)
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
