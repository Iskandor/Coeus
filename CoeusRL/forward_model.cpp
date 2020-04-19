#include "forward_model.h"


forward_model::forward_model(neural_network* p_network, optimizer* p_optimizer) :
	_network(p_network),
	_optimizer(p_optimizer)
{
}

forward_model::~forward_model()
{
}

void forward_model::train(tensor* p_state, tensor* p_action, tensor* p_next_state)
{
	std::vector<tensor*> s0a;
	s0a.push_back(p_state);
	s0a.push_back(p_action);
	tensor::concat(s0a, _input, 0);

	tensor& predicted_state = _network->forward(&_input);
	_network->backward(_loss_function.backward(predicted_state, *p_next_state));
	_optimizer->update();
}

tensor& forward_model::reward(tensor* p_state, tensor* p_action, tensor* p_next_state)
{
	std::vector<tensor*> s0a;
	s0a.push_back(p_state);
	s0a.push_back(p_action);
	tensor::concat(s0a, _input, 0);

	tensor& predicted_state = _network->forward(&_input);

	_reward.resize({ predicted_state.shape(0), 1 });

	for(int i = 0; i < predicted_state.shape(0); i++)
	{
		for(int j = 0; j < predicted_state.shape(1); j++)
		{
			const int index = i * predicted_state.shape(1) + j;
			_reward[i] += pow(predicted_state[index] - (*p_next_state)[index], 2);
		}
		_reward[i] *= 0.5f;
	}

	return _reward;
}
