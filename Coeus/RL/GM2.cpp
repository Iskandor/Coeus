#include "GM2.h"
#include "RuleFactory.h"
#include "QuadraticCost.h"

using namespace Coeus;

GM2::GM2(NeuralNetwork* p_autoencoder, GRADIENT_RULE p_rule, float p_alpha, int p_size)
{
	_autoencoder = p_autoencoder;
	_gradient = new NetworkGradient(p_autoencoder);
	_rule = RuleFactory::create_rule(p_rule, p_autoencoder, p_alpha);

	_buffer = nullptr;

	if (p_size > 0)
	{
		_buffer = new ReplayBuffer<TransitionItem>(p_size);
	}
}

GM2::~GM2()
{
	delete _gradient;
	delete _rule;
	delete _buffer;
}

void GM2::add(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) const
{
	if (_buffer != nullptr)
	{
		_buffer->add_item(new TransitionItem(p_state0, p_action, p_state1));
	}
}

void GM2::activate(Tensor* p_state) const
{
	_autoencoder->activate(p_state);
}

float GM2::train(Tensor* p_state)
{
	activate(p_state);
	const float error = _mse.cost(_autoencoder->get_output(), p_state);
	Tensor loss = _mse.cost_deriv(_autoencoder->get_output(), p_state);

	_gradient->calc_gradient(&loss);
	_rule->calc_update(_gradient->get_gradient());
	_autoencoder->update(_rule->get_update());
	
	return error;
}

float GM2::train(const int p_sample)
{
	float error = 0;

	if (_buffer->get_size() >= p_sample) {
		vector<TransitionItem*>* sample = _buffer->get_sample(p_sample);
		
		Tensor input({ p_sample, _autoencoder->get_input_dim() }, Tensor::ZERO);

		for(auto s : *sample)
		{
			input.insert_row(&s->s0);
		}

		error = train(&input);
	}

	return error;
}

float GM2::uncertainty_motivation(Tensor* p_state, const float p_eta)
{
	activate(p_state);
	_autoencoder->get_output()->reshape({ 4,4 });
	cout << *_autoencoder->get_output() << endl;
	p_state->reshape({ 4,4 });
	cout << *p_state << endl;
	const float intrinsic_reward = p_state->size() *  p_eta * _mse.cost(_autoencoder->get_output(), p_state);
	return intrinsic_reward;
}
