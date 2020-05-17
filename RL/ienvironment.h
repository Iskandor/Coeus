#pragma once
#include "tensor.h"

class COEUS_DLL_API ienvironment
{
public:
	ienvironment()
	= default;

	ienvironment(ienvironment& p_copy);
	ienvironment& operator = (const ienvironment& p_copy);

	virtual ~ienvironment()
	= default;

	virtual tensor	get_state() = 0;
	virtual void	do_action(tensor& p_action) = 0;
	virtual float	get_reward() = 0;
	virtual void	reset() = 0;
	virtual bool	is_finished() = 0;

	int STATE_DIM() const { return _state_dim; }
	int ACTION_DIM() const { return _action_dim; }

protected:
	int _state_dim{};
	int _action_dim{};
};

inline ienvironment::ienvironment(ienvironment& p_copy)
{
	_state_dim = p_copy._state_dim;
	_action_dim = p_copy._action_dim;
}

inline ienvironment& ienvironment::operator=(const ienvironment& p_copy)
{
	_state_dim = p_copy._state_dim;
	_action_dim = p_copy._action_dim;
	return *this;
}
