#pragma once
class SimpleContinuousEnv
{
public:
	SimpleContinuousEnv();
	~SimpleContinuousEnv();

	float get_state() const;
	void perform_action(float p_action);
	float get_reward() const;
	bool is_finished() const;
	bool is_winner() const;
	bool is_failed() const;
	void reset();

private:
	float _position;
	float _target;
	int _winning_position;
	const float THETA = 0.5f;
};

