#include "ICM.h"
#include "Encoder.h"
#include "RuleFactory.h"
#include "NeuronOperator.h"

using namespace Coeus;

ICM::ICM(NeuralNetwork* p_forward_model, NeuralNetwork* p_inverse_model, NeuralNetwork* p_head, GRADIENT_RULE p_rule, float p_alpha, const int p_size)
{
	_forward_model = p_forward_model;
	_inverse_model = p_inverse_model;
	_head = p_head;

	_fm_gradient = new NetworkGradient(p_forward_model);
	_im_gradient = new NetworkGradient(p_inverse_model);
	_h_gradient = new NetworkGradient(p_head);

	_fm_rule = RuleFactory::create_rule(p_rule, p_forward_model, p_alpha);
	_im_rule = RuleFactory::create_rule(p_rule, p_inverse_model, p_alpha);
	_h_rule = RuleFactory::create_rule(p_rule, p_head, p_alpha);
	
	_forward_reward = 0;
	_buffer = nullptr;

	if (p_size > 0)
	{
		_buffer = new ReplayBuffer<TransitionItem>(p_size);
	}

	_fm_input = nullptr;
	_im_input = nullptr;
	_fm_target = nullptr;
	_im_target = nullptr;
	_h_input_s0 = nullptr;
	_h_input_s1 = nullptr;
}

ICM::~ICM()
{
	delete _buffer;
	delete _fm_input;
	delete _im_input;
	delete _fm_target;
	delete _im_target;
	delete _h_input_s0;
	delete _h_input_s1;
}

void ICM::activate(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, bool p_inverse_model)
{
	if (_fm_input == nullptr || _fm_input->size() != _forward_model->get_input_dim()) {
		delete _fm_input;
		_fm_input = new Tensor({ _forward_model->get_input_dim() }, Tensor::ZERO);
	}

	if (_im_input == nullptr || _im_input->size() != _inverse_model->get_input_dim()) {
		delete _im_input;
		_im_input = new Tensor({ _inverse_model->get_input_dim() }, Tensor::ZERO);
	}

	_head->activate(p_state0);
		
	_fm_input->reset_index();
	_fm_input->push_back(_head->get_output());
	_fm_input->push_back(p_action);
	
	_forward_model->activate(_fm_input);

	if (p_inverse_model)
	{
		_im_input->reset_index();
		_im_input->push_back(_head->get_output());
		_head->activate(p_state1);
		_im_input->push_back(_head->get_output());

		_inverse_model->activate(_im_input);
	}	
}

float ICM::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) {

	float error = 0;
	
	activate(p_state0, p_action, p_state1, true);

	QuadraticCost L;
	Gradient hg;

	Tensor fm_loss = L.cost_deriv(_forward_model->get_output(), p_state1);
	error += L.cost(_forward_model->get_output(), p_state1);
	Tensor im_loss = L.cost_deriv(_inverse_model->get_output(), p_action);
	error += L.cost(_inverse_model->get_output(), p_action);

	_fm_gradient->calc_gradient(&fm_loss);
	_im_gradient->calc_gradient(&im_loss);
	_head->activate(p_state0);
	
	Tensor h_loss = _fm_gradient->get_input_gradient(1, 0, _head->get_output_dim());
	_h_gradient->calc_gradient(&h_loss);
	hg = _h_gradient->get_gradient();

	h_loss = _im_gradient->get_input_gradient(1, 0, _head->get_output_dim());
	_h_gradient->calc_gradient(&h_loss);
	hg += _h_gradient->get_gradient();
	
	_head->activate(p_state1);

	h_loss = _im_gradient->get_input_gradient(1, _head->get_output_dim(), _head->get_output_dim());
	_h_gradient->calc_gradient(&h_loss);
	hg += _h_gradient->get_gradient();

	_fm_rule->calc_update(_fm_gradient->get_gradient());
	_im_rule->calc_update(_im_gradient->get_gradient());
	_h_rule->calc_update(hg);

	_forward_model->update(_fm_rule->get_update());
	_inverse_model->update(_im_rule->get_update());
	_head->update(_h_rule->get_update());

	return error;
}

void ICM::add(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) const
{
	if (_buffer != nullptr)
	{
		_buffer->add_item(new TransitionItem(p_state0, p_action, p_state1));
	}

}

float ICM::train(const int p_sample)
{
	float error = 0;

	if (_buffer->get_size() >= p_sample) {
		_fm_input = NeuronOperator::init_auxiliary_parameter(_fm_input, p_sample, _forward_model->get_input_dim());
		_im_input = NeuronOperator::init_auxiliary_parameter(_im_input, p_sample, _inverse_model->get_input_dim());
		_fm_target = NeuronOperator::init_auxiliary_parameter(_fm_target, p_sample, _forward_model->get_output_dim());
		_im_target = NeuronOperator::init_auxiliary_parameter(_im_target, p_sample, _inverse_model->get_output_dim());
		_h_input_s0 = NeuronOperator::init_auxiliary_parameter(_h_input_s0, p_sample, _head->get_input_dim());
		_h_input_s1 = NeuronOperator::init_auxiliary_parameter(_h_input_s1, p_sample, _head->get_input_dim());

		vector<TransitionItem*>* sample = _buffer->get_sample(p_sample);

		_fm_target->reset_index();
		_im_target->reset_index();
		_h_input_s0->reset_index();
		_h_input_s1->reset_index();

		for (auto& s : *sample)
		{
			//cout << s->s0 << ":" << s->a << endl;
			_h_input_s0->insert_row(&s->s0);
			_h_input_s1->insert_row(&s->s1);
			_fm_target->insert_row(&s->s1);
			_im_target->insert_row(&s->a);
		}
		//cout << endl;

		_head->activate(_h_input_s0);

		_fm_input->reset_index();
		_fm_input->insert_column(_head->get_output());
		_fm_input->insert_column(_im_target);

		_forward_model->activate(_fm_input);

		_im_input->reset_index();
		_im_input->insert_column(_head->get_output());
		_head->activate(_h_input_s1);
		_im_input->insert_column(_head->get_output());

		_inverse_model->activate(_im_input);

		QuadraticCost L;
		Gradient hg;

		Tensor fm_loss = L.cost_deriv(_forward_model->get_output(), _fm_target);
		error += L.cost(_forward_model->get_output(), _fm_target);
		Tensor im_loss = L.cost_deriv(_inverse_model->get_output(), _im_target);
		error += L.cost(_inverse_model->get_output(), _im_target);

		_fm_gradient->calc_gradient(&fm_loss);
		_im_gradient->calc_gradient(&im_loss);
		_head->activate(_h_input_s0);

		Tensor h_loss = _fm_gradient->get_input_gradient(p_sample, 0, _head->get_output_dim());
		_h_gradient->calc_gradient(&h_loss);
		hg = _h_gradient->get_gradient();

		h_loss = _im_gradient->get_input_gradient(p_sample, 0, _head->get_output_dim());
		_h_gradient->calc_gradient(&h_loss);
		hg += _h_gradient->get_gradient();

		_head->activate(_h_input_s1);

		h_loss = _im_gradient->get_input_gradient(p_sample, _head->get_output_dim(), _head->get_output_dim());
		_h_gradient->calc_gradient(&h_loss);
		hg += _h_gradient->get_gradient();

		_fm_rule->calc_update(_fm_gradient->get_gradient());
		_im_rule->calc_update(_im_gradient->get_gradient());
		_h_rule->calc_update(hg);

		_forward_model->update(_fm_rule->get_update());
		_inverse_model->update(_im_rule->get_update());
		_head->update(_h_rule->get_update());
	}

	return error;
}

float ICM::get_intrinsic_reward(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, const float p_eta) {
	activate(p_state0, p_action, p_state1);
	_forward_reward = _L.cost(_forward_model->get_output(), p_state1);
	return p_eta *  p_state1->size() * _forward_reward;

	/* (1 - _forward_reward)/_forward_reward */
}

