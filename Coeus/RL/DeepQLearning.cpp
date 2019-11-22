#include "DeepQLearning.h"
#include "QuadraticCost.h"
#include "CoreLayer.h"

using namespace Coeus;

DeepQLearning::DeepQLearning(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma,	int p_size, int p_sample, int p_target_network_update) : QLearning(p_network, p_grad_rule, p_alpha, p_gamma)
{
	_replay_buffer = new ReplayBuffer<DQItem>(p_size);
	_sample_size = p_sample;

	_target = Tensor::Zero({ p_sample, _network->get_output_dim() });
	_input_s0 = Tensor::Zero({ p_sample, _network->get_input_dim() });
	_input_s1 = Tensor::Zero({ p_sample, _network->get_input_dim() });
	_max_qa = Tensor::Zero({ p_sample });

	_target_network = new NeuralNetwork(*p_network, true);
	_target_network_update = p_target_network_update;
	_target_network_update_t = 0;
}

DeepQLearning::~DeepQLearning()
{
	delete _replay_buffer;
	delete _target_network;
}

float DeepQLearning::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const float p_reward, const bool p_final) {
	Tensor action({ 1 }, Tensor::VALUE, p_action0);
	_replay_buffer->add_item(new DQItem(p_state0, &action, p_state1, p_reward, p_final));

	float error = 0;

	if (_replay_buffer->get_size() >= _sample_size) {
		_target_network_update_t++;
		vector<DQItem*>* sample = _replay_buffer->get_sample(_sample_size);

		_input_s0.reset_index();
		_input_s1.reset_index();
		for (auto& s : *sample)
		{
			_input_s0.push_back(&s->s0);
			_input_s1.push_back(&s->s1);
		}

		_network->activate(&_input_s0);
		_target = *_network->get_output();

		calc_max_qa(_sample_size);

		for (int i = 0; i < sample->size(); i++) {
			if (sample->at(i)->final) {
				_target.set(i, sample->at(i)->a[0], sample->at(i)->r);
			}
			else {
				_target.set(i, sample->at(i)->a[0], sample->at(i)->r + _gamma * _max_qa[i]);
			}
		}

		QuadraticCost mse;

		Tensor loss = mse.cost_deriv(_network->get_output(), &_target);

		_network_gradient->calc_gradient(&loss);
		_update_rule->calc_update(_network_gradient->get_gradient(), _alpha);
		_network->update(_update_rule->get_update());

		if (_target_network_update_t == _target_network_update)
		{
			_target_network_update_t = 0;
			_target_network->copy_params(_network);
		}
	}

	return error;
}

void DeepQLearning::calc_max_qa(const int p_sample)
{
	_target_network->activate(&_input_s1);
	Tensor* output = _target_network->get_output();

	for(int i = 0; i < p_sample; i++)
	{
		float maxQ = 0;
		for(int j = 0; j < _network->get_output_dim(); j++)
		{
			if (maxQ < output->at(i,j))
			{
				maxQ = output->at(i, j);
			}
		}
		_max_qa[i] = maxQ;
	}
}

float DeepQLearning::calc_max_qa(Tensor* p_state) const {
	
	const int maxQa = _target_network->get_output()->max_value_index();

	return _target_network->get_output()->at(maxQa);
}
