#pragma once
#include "Tensor.h"

namespace Coeus
{
	class __declspec(dllexport) IEnvironment
	{
	public:
		IEnvironment()
		= default;

		IEnvironment(IEnvironment& p_copy);
		IEnvironment& operator = (const IEnvironment& p_copy);

		virtual ~IEnvironment()
		= default;

		virtual Tensor	get_state() = 0;
		virtual void	do_action(Tensor& p_action) = 0;
		virtual float	get_reward() = 0;
		virtual void	reset() = 0;
		virtual bool	is_finished() = 0;

		int STATE_DIM() const { return _state_dim; }
		int ACTION_DIM() const { return _action_dim; }

	protected:
		int _state_dim{};
		int _action_dim{};
	};

	inline IEnvironment::IEnvironment(IEnvironment& p_copy)
	{
		_state_dim = p_copy._state_dim;
		_action_dim = p_copy._action_dim;
	}

	inline IEnvironment& IEnvironment::operator=(const IEnvironment& p_copy)
	{
		_state_dim = p_copy._state_dim;
		_action_dim = p_copy._action_dim;
		return *this;
	}
}
