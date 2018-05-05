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

void RecurrentLayerGradient::calc_delta(Tensor* p_weights, Tensor* p_delta) {
}

void RecurrentLayerGradient::calc_gradient(map<string, Tensor>& p_gradient) {
}
