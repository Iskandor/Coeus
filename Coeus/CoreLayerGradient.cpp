#include "CoreLayerGradient.h"

using namespace Coeus;

CoreLayerGradient::CoreLayerGradient(BaseLayer* p_layer, NeuralNetwork* p_network) : IGradientComponent(p_layer, p_network) {
}

CoreLayerGradient::~CoreLayerGradient()
{
}

void CoreLayerGradient::init()
{
	BaseCellGroup* g = get_layer<CoreLayer>()->_output_group;
	_deriv[g->get_id()] = Tensor::Zero({ g->get_dim() });
	_delta[g->get_id()] = Tensor::Zero({ g->get_dim() });
}

void CoreLayerGradient::calc_deriv() {
	calc_deriv_group(get_layer<CoreLayer>()->_output_group);
}

void CoreLayerGradient::calc_delta(Tensor* p_weights, Tensor* p_delta) {
	BaseCellGroup* g = get_layer<CoreLayer>()->_output_group;
	_delta[g->get_id()] = _deriv[g->get_id()] * (p_weights->T() * *p_delta);
}

void CoreLayerGradient::calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) {
	BaseCellGroup* g = get_layer<CoreLayer>()->_output_group;
	vector<BaseLayer*> input_layers = _network->get_input_layers(_layer->get_id());

	for (auto it = input_layers.begin(); it != input_layers.end(); ++it) {
		Connection* c = _network->get_connection((*it)->get_id(), _layer->get_id());
		if (c->is_trainable())
		{
			p_w_gradient[c->get_id()] = *_network->get_layer((*it)->get_id())->get_output() * _delta[get_layer<CoreLayer>()->_input_group->get_id()];
		}
	}

	p_b_gradient[g->get_id()] = _delta[g->get_id()];
}
