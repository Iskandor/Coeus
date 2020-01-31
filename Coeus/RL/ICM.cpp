#include "ICM.h"
#include "Encoder.h"
#include "RuleFactory.h"
#include "NeuronOperator.h"

using namespace Coeus;

ICM::ICM(NeuralNetwork* p_forward_model, NeuralNetwork* p_inverse_model, NeuralNetwork* p_feature_extractor, GRADIENT_RULE p_rule, float p_alpha, const int p_size, float p_beta)
{
	_forward_model = p_forward_model;
	_inverse_model = p_inverse_model;
	_feature_extractor = p_feature_extractor;

	_fm_gradient = new NetworkGradient(p_forward_model);
	_im_gradient = new NetworkGradient(p_inverse_model);
	_fe_gradient = new NetworkGradient(p_feature_extractor);

	_fm_rule = RuleFactory::create_rule(p_rule, p_forward_model, p_alpha);
	_im_rule = RuleFactory::create_rule(p_rule, p_inverse_model, p_alpha);
	_fe_rule = RuleFactory::create_rule(p_rule, p_feature_extractor, p_alpha);

	_beta = p_beta;
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
	_fe_input_s0 = nullptr;
	_fe_input_s1 = nullptr;
}

ICM::~ICM()
{
	delete _buffer;
	delete _fm_input;
	delete _im_input;
	delete _fm_target;
	delete _im_target;
	delete _fe_input_s0;
	delete _fe_input_s1;
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

	_feature_extractor->activate(p_state0);
		
	_fm_input->reset_index();
	_fm_input->push_back(_feature_extractor->get_output());
	_fm_input->push_back(p_action);
	
	_forward_model->activate(_fm_input);

	if (p_inverse_model)
	{
		_im_input->reset_index();
		_im_input->push_back(_feature_extractor->get_output());
		_feature_extractor->activate(p_state1);
		_im_input->push_back(_feature_extractor->get_output());

		_inverse_model->activate(_im_input);
	}	
}

float ICM::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) {

	float error = 0;
	
	activate(p_state0, p_action, p_state1, true);

	Gradient fe_gradient;

	Tensor fm_loss = _L.cost_deriv(_forward_model->get_output(), p_state1) * _beta;
	error += _L.cost(_forward_model->get_output(), p_state1);
	Tensor im_loss = _L.cost_deriv(_inverse_model->get_output(), p_action) * (1 - _beta);
	error += _L.cost(_inverse_model->get_output(), p_action);

	_fm_gradient->calc_gradient(&fm_loss);
	_im_gradient->calc_gradient(&im_loss);
	_feature_extractor->activate(p_state0);
	
	Tensor fe_loss = _fm_gradient->get_input_gradient(1, 0, _feature_extractor->get_output_dim());
	_fe_gradient->calc_gradient(&fe_loss);
	fe_gradient = _fe_gradient->get_gradient();

	fe_loss = _im_gradient->get_input_gradient(1, 0, _feature_extractor->get_output_dim());
	_fe_gradient->calc_gradient(&fe_loss);
	fe_gradient += _fe_gradient->get_gradient();
	
	_feature_extractor->activate(p_state1);

	fe_loss = _im_gradient->get_input_gradient(1, _feature_extractor->get_output_dim(), _feature_extractor->get_output_dim());
	_fe_gradient->calc_gradient(&fe_loss);
	fe_gradient += _fe_gradient->get_gradient();

	_fm_rule->calc_update(_fm_gradient->get_gradient());
	_im_rule->calc_update(_im_gradient->get_gradient());
	_fe_rule->calc_update(fe_gradient);

	_forward_model->update(_fm_rule->get_update());
	_inverse_model->update(_im_rule->get_update());
	_feature_extractor->update(_fe_rule->get_update());

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
		_fe_input_s0 = NeuronOperator::init_auxiliary_parameter(_fe_input_s0, p_sample, _feature_extractor->get_input_dim());
		_fe_input_s1 = NeuronOperator::init_auxiliary_parameter(_fe_input_s1, p_sample, _feature_extractor->get_input_dim());

		vector<TransitionItem*>* sample = _buffer->get_sample(p_sample);

		_fm_target->reset_index();
		_im_target->reset_index();
		_fe_input_s0->reset_index();
		_fe_input_s1->reset_index();

		for (auto& s : *sample)
		{
			//cout << s->s0 << ":" << s->a << endl;
			_fe_input_s0->insert_row(&s->s0);
			_fe_input_s1->insert_row(&s->s1);
			_fm_target->insert_row(&s->s1);
			_im_target->insert_row(&s->a);
		}
		//cout << endl;

		_feature_extractor->activate(_fe_input_s0);

		_fm_input->reset_index();
		_fm_input->insert_column(_feature_extractor->get_output());
		_fm_input->insert_column(_im_target);

		_forward_model->activate(_fm_input);

		_im_input->reset_index();
		_im_input->insert_column(_feature_extractor->get_output());
		_feature_extractor->activate(_fe_input_s1);
		_im_input->insert_column(_feature_extractor->get_output());

		_inverse_model->activate(_im_input);

		Gradient fe_gradient;

		Tensor fm_loss = _L.cost_deriv(_forward_model->get_output(), _fm_target) * _beta;
		error += _L.cost(_forward_model->get_output(), _fm_target);
		Tensor im_loss = _L.cost_deriv(_inverse_model->get_output(), _im_target) * (1 - _beta);
		error += _L.cost(_inverse_model->get_output(), _im_target);

		_fm_gradient->calc_gradient(&fm_loss);
		_im_gradient->calc_gradient(&im_loss);
		_feature_extractor->activate(_fe_input_s0);

		Tensor fe_loss = _fm_gradient->get_input_gradient(p_sample, 0, _feature_extractor->get_output_dim());
		_fe_gradient->calc_gradient(&fe_loss);
		fe_gradient = _fe_gradient->get_gradient();

		fe_loss = _im_gradient->get_input_gradient(p_sample, 0, _feature_extractor->get_output_dim());
		_fe_gradient->calc_gradient(&fe_loss);
		fe_gradient += _fe_gradient->get_gradient();

		_feature_extractor->activate(_fe_input_s1);

		fe_loss = _im_gradient->get_input_gradient(p_sample, _feature_extractor->get_output_dim(), _feature_extractor->get_output_dim());
		_fe_gradient->calc_gradient(&fe_loss);
		fe_gradient += _fe_gradient->get_gradient();

		_fm_rule->calc_update(_fm_gradient->get_gradient());
		_im_rule->calc_update(_im_gradient->get_gradient());
		_fe_rule->calc_update(fe_gradient);

		_forward_model->update(_fm_rule->get_update());
		_inverse_model->update(_im_rule->get_update());
		_feature_extractor->update(_fe_rule->get_update());
	}

	return error;
}

float ICM::get_intrinsic_reward(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, const float p_eta) {
	activate(p_state0, p_action, p_state1);
	_forward_reward = _L.cost(_forward_model->get_output(), p_state1);
	return p_eta *  p_state1->size() * _forward_reward;
}

