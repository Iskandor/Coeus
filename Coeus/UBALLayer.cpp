#include "UBALLayer.h"

using namespace Coeus;

UBALLayer::UBALLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation, BaseLayer* p_layer) : BaseLayer(p_id)
{
	_input_group = p_layer->get_output_group();
	_input_bp = new NeuralGroup(_input_group->get_dim(), p_activation, true);
	_input_bp->set_bias(_input_group->get_bias());
	_input_echo = new NeuralGroup(_input_group->get_dim(), p_activation, true);
	_input_echo->set_bias(_input_group->get_bias());

	_output_group = new NeuralGroup(p_dim, p_activation, true);
	_output_bp = new NeuralGroup(p_dim, p_activation, true);
	_output_bp->set_bias(_output_group->get_bias());
	_output_echo = new NeuralGroup(p_dim, p_activation, true);
	_output_echo->set_bias(_output_group->get_bias());

	_forward_W = add_connection(new Connection(_input_group->get_dim(), _output_group->get_dim(), _input_group->get_id(), _output_group->get_id()));
	_backward_M = add_connection(new Connection(_output_group->get_dim(), _input_group->get_dim(), _output_group->get_id(), _input_group->get_id()));
}


UBALLayer::~UBALLayer()
{
	delete _output_group;
	delete _input_bp;
	delete _output_bp;
	delete _input_echo;
	delete _output_echo;
	delete _forward_W;
	delete _backward_M;
}

void UBALLayer::integrate(Tensor * p_input, Tensor * p_weights)
{
}

void UBALLayer::activate(Tensor * p_input)
{
	_output_group->integrate(_input_group->get_output(), _forward_W->get_weights());
	_output_group->activate();
	_input_echo->integrate(_output_group->get_output(), _backward_M->get_weights());
	_input_echo->activate();
}

void UBALLayer::activate_back(Tensor * p_input)
{
	if (p_input != nullptr) {
		_output_bp->set_output(p_input);
	}

	_input_bp->integrate(_output_bp->get_output(), _backward_M->get_weights());
	_input_bp->activate();
	_output_echo->integrate(_input_bp->get_output(), _forward_W->get_weights());
	_output_echo->activate();
}

void UBALLayer::override_params(BaseLayer * p_source)
{

}