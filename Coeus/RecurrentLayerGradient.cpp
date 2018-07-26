#include "RecurrentLayerGradient.h"

using namespace Coeus;

RecurrentLayerGradient::RecurrentLayerGradient(RecurrentLayer* p_recurrent_layer) : IGradientComponent(p_recurrent_layer)
{

}


RecurrentLayerGradient::~RecurrentLayerGradient()
{
}

void RecurrentLayerGradient::init() {
}

void RecurrentLayerGradient::calc_deriv() {
}

void RecurrentLayerGradient::calc_delta(Tensor* p_weights, LayerState* p_state) {
}

void RecurrentLayerGradient::calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) {
}

void RecurrentLayerGradient::set_delta(Tensor* p_delta)
{
}
