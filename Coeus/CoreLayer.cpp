#include "CoreLayer.h"
#include "CoreLayerGradient.h"
#include "IDGen.h"

using namespace Coeus;

CoreLayer::CoreLayer(const string& p_id, const int p_dim, const ACTIVATION p_activation) : BaseLayer(p_id)
{
	_type = CORE;
	_group = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, p_activation, true));
	_input_group = _output_group = _group;
}

CoreLayer::CoreLayer(json p_data) : BaseLayer(p_data)
{
	_type = CORE;
	_group = add_group<SimpleCellGroup>(new SimpleCellGroup(p_data["group"]));
	_input_group = _output_group = _group;
}

CoreLayer::CoreLayer(CoreLayer &p_copy) : BaseLayer(IDGen::instance().next()) {
	_type = CORE;
	_group = add_group<SimpleCellGroup>(new SimpleCellGroup(*p_copy._group));
	_input_group = _output_group = _group;
}

CoreLayer::~CoreLayer()
{
	delete _group;
}

void CoreLayer::integrate(Tensor* p_input, Tensor* p_weights) {
	_group->integrate(p_input, p_weights);
}

void CoreLayer::activate(Tensor * p_input)
{	
	_group->activate();
}

void CoreLayer::override(BaseLayer * p_source)
{
	CoreLayer* source = dynamic_cast<CoreLayer*>(p_source);
	_group->get_bias()->override(source->_group->get_bias());
}

json CoreLayer::get_json() const
{
	json data = BaseLayer::get_json();

	data["group"] = _group->get_json();

	return data;
}

CoreLayer::CoreLayer(CoreLayer* p_source) : BaseLayer(p_source)
{
	_type = CORE;
	_group = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_group));
	_input_group = _output_group = _group;
}

CoreLayer* CoreLayer::clone()
{
	return new CoreLayer(this);
}
