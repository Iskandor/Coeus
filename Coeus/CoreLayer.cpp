#include "CoreLayer.h"
#include "CoreLayerGradient.h"
#include "IDGen.h"

using namespace Coeus;

CoreLayer::CoreLayer(const string p_id, const int p_dim, const ACTIVATION p_activation) : BaseLayer(p_id)
{
	_output_group = add_group(new SimpleCellGroup(p_dim, p_activation, true));
	_input_group = _output_group;

	_type = CORE;
	_gradient_component = new CoreLayerGradient(this);
}

CoreLayer::CoreLayer(CoreLayer &p_copy) : BaseLayer(IDGen::instance().next()) {
	_output_group = add_group(new SimpleCellGroup(*p_copy._output_group));
	_input_group = _output_group;

	_type = CORE;
	_gradient_component = new CoreLayerGradient(this);
}

CoreLayer::~CoreLayer()
{
	delete _output_group;
	delete _gradient_component;
}

void CoreLayer::integrate(Tensor* p_input, Tensor* p_weights) {
	_output_group->integrate(p_input, p_weights);
}

void CoreLayer::activate(Tensor * p_input)
{	
	_output_group->activate();
}

void CoreLayer::override(BaseLayer * p_source)
{
	CoreLayer* source = dynamic_cast<CoreLayer*>(p_source);
	_output_group->get_bias()->override(source->_output_group->get_bias());
}