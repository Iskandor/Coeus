#include "metacritic.h"

#include <algorithm>

metacritic::metacritic(neural_network* p_network, optimizer* p_optimizer, forward_model* p_forward_model, float p_sigma) :
	_network(p_network),
	_optimizer(p_optimizer),
	_forward_model(p_forward_model),
	_sigma(p_sigma)
{
}

metacritic::~metacritic()
{
}

void metacritic::train(tensor* p_state, tensor* p_action, tensor* p_next_state)
{
	std::vector<tensor*> s0a;
	s0a.push_back(p_state);
	s0a.push_back(p_action);
	tensor::concat(s0a, _input, 0);
	
	tensor& target = _forward_model->error(p_state, p_action, p_next_state);
	tensor& prediction = _network->forward(&_input);

	_network->backward(_loss_function.backward(prediction, target));
	_optimizer->update();

	_forward_model->train(p_state, p_action, p_next_state);
}

tensor& metacritic::reward(tensor* p_state, tensor* p_action, tensor* p_next_state)
{
	std::vector<tensor*> s0a;
	s0a.push_back(p_state);
	s0a.push_back(p_action);
	tensor::concat(s0a, _input, 0);

	tensor& error = _forward_model->error(p_state, p_action, p_next_state);
	tensor& error_estimate = _network->forward(&_input);
	tensor& pe_reward = _forward_model->reward(p_state, p_action, p_next_state);

	_reward.resize({ error.size() });

	for(int i = 0; i < _reward.size(); i++)
	{
		if (abs(error[i] - error_estimate[i]) > _sigma)
		{
			_reward[i] = std::max(tanh(error[i] / error_estimate[i] + error_estimate[i] / error[i] - 2), 0.f);
		}
		else
		{
			_reward[i] = 0.f;
		}
		_reward[i] = std::max(pe_reward[i], _reward[i]);
	}
	return _reward;
}

tensor& metacritic::error(tensor* p_state, tensor* p_action)
{
	std::vector<tensor*> s0a;
	s0a.push_back(p_state);
	s0a.push_back(p_action);
	tensor::concat(s0a, _input, 0);

	_error = _network->forward(&_input);
	
	return _error;
}
