#include "RecurrentLayerGradient.h"

using namespace Coeus;

RecurrentLayerGradient::RecurrentLayerGradient(BaseLayer* p_layer, NeuralNetwork* p_network) : IGradientComponent(p_layer, p_network)
{

}


RecurrentLayerGradient::~RecurrentLayerGradient()
{
}

void RecurrentLayerGradient::init() {
}

void RecurrentLayerGradient::calc_deriv() {
}

void RecurrentLayerGradient::calc_delta(Tensor* p_weights, Tensor* p_delta) {
}

void RecurrentLayerGradient::calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) {
}
