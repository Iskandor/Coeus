//
// Created by mpechac on 7. 3. 2017.
//

#ifndef NEURONET_MAZE_H
#define NEURONET_MAZE_H

#include <vector>
#include <ienvironment.h>

using namespace std;

class maze : public ienvironment {
public:
    maze();
	maze(maze& p_copy);
	maze(int* p_topology, unsigned int p_mazeX, unsigned int p_mazeY, int p_goal, bool p_stochastic = false);
    ~maze();

	maze& operator = (const maze& p_copy);

	void init(int* p_topology, unsigned int p_mazeX, unsigned int p_mazeY, int p_goal, bool p_stochastic = false);

	tensor get_state() override;
	void set_state(tensor& p_state);
	void do_action(tensor& p_action) override;
	float get_reward() override;
	bool is_finished() override;
	void reset() override;

	bool is_winner() const;
	
    string to_string();
	string to_string(int p_row);

    int actor() const {
        return _actor;
    }

    int goal() const {
        return _goal;
    }

	int moves() const {
		return _steps;
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
    int move(int p_x, int p_y) const;
	
    unsigned int _mazeX{}, _mazeY{};
    vector<int> _init_pos;
    vector<int> _maze_table;

    int _actor{};
    int _goal{};
    bool _bang{};
    bool _kill{};

	int _steps{};
	bool _stochastic{};

	int* _topology{};

	const float defautPenalty = 0;
	const float bangPenalty = 0;
	const float killPenalty = 0;
	const float finalReward = 1;
};

#endif //NEURONET_MAZE_H
