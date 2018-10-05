#include "LSTMLayerGradient.h"
#include "ActivationFunctionsDeriv.h"
#include "LSTMLayerState.h"

using namespace Coeus;

LSTMLayerGradient::LSTMLayerGradient(LSTMLayer* p_layer) : IGradientComponent(p_layer)
{

}


LSTMLayerGradient::~LSTMLayerGradient()
{
}

void LSTMLayerGradient::init()
{

}

void LSTMLayerGradient::calc_deriv()
{

}

void LSTMLayerGradient::calc_delta(Tensor* p_weights, Tensor* p_delta)
{

}

void LSTMLayerGradient::calc_gradient(map<string, Tensor>& p_w_gradient, map<string, Tensor>& p_b_gradient)
{

}
