#include "CoreLayerGradient.h"

using namespace Coeus;

CoreLayerGradient::CoreLayerGradient()
{
}


CoreLayerGradient::~CoreLayerGradient()
{
}

void Coeus::CoreLayerGradient::init(BaseLayer * p_layer)
{
}

map<string, Tensor>* Coeus::CoreLayerGradient::calc_delta(Tensor * p_delta)
{
	return nullptr;
}

map<string, Tensor>* Coeus::CoreLayerGradient::calc_gradient()
{
	return nullptr;
}
