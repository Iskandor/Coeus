#include "LSTMCellGroup.h"
#include "IOUtils.h"

using namespace Coeus;

LSTMCellGroup::LSTMCellGroup(int p_dim, ACTIVATION p_activation_function, SimpleCellGroup* p_input_gate, SimpleCellGroup* p_output_gate, SimpleCellGroup* p_forget_gate) : BaseCellGroup(p_dim, true)
{
	_state = Tensor::Zero({p_dim});
	_g_output = Tensor::Zero({ p_dim });
	_h_output = Tensor::Zero({ p_dim });
	_output = Tensor::Zero({ p_dim });

	_f = init_activation_function(p_activation_function);
	_g = init_activation_function(TANH);

	_input_gate = p_input_gate;
	_output_gate = p_output_gate;
	_forget_gate = p_forget_gate;
}

LSTMCellGroup::LSTMCellGroup(json p_data, SimpleCellGroup* p_input_gate, SimpleCellGroup* p_output_gate, SimpleCellGroup*p_forget_gate) : BaseCellGroup(p_data)
{
	_state = Tensor::Zero({ _dim });
	_g_output = Tensor::Zero({ _dim });
	_h_output = Tensor::Zero({ _dim });
	_output = Tensor::Zero({ _dim });

	_f = IOUtils::init_activation_function(p_data["f"]);
	_g = IOUtils::init_activation_function(p_data["g"]);

	_input_gate = p_input_gate;
	_output_gate = p_output_gate;
	_forget_gate = p_forget_gate;
}

LSTMCellGroup::LSTMCellGroup(LSTMCellGroup& p_copy) : BaseCellGroup(p_copy._dim, p_copy._bias_flag)
{
	_state = Tensor::Zero({ p_copy._dim });
	_g_output = Tensor::Zero({ p_copy._dim });
	_h_output = Tensor::Zero({ p_copy._dim });
	_output = Tensor::Zero({ p_copy._dim });

	_f = init_activation_function(p_copy._f->get_type());
	_g = init_activation_function(TANH);
	_input_gate = p_copy._input_gate;
	_output_gate = p_copy._output_gate;
	_forget_gate = p_copy._forget_gate;

}

LSTMCellGroup& LSTMCellGroup::operator=(const LSTMCellGroup& p_copy)
{
	copy(p_copy);

	_state = Tensor::Zero({ p_copy._dim });
	_g_output = Tensor::Zero({ p_copy._dim });
	_h_output = Tensor::Zero({ p_copy._dim });
	_output = Tensor::Zero({ p_copy._dim });

	_f = init_activation_function(p_copy._f->get_type());
	_g = init_activation_function(TANH);
	_input_gate = p_copy._input_gate;
	_output_gate = p_copy._output_gate;
	_forget_gate = p_copy._forget_gate;

	return *this;
}

LSTMCellGroup::~LSTMCellGroup()
{
	delete _g;
}

void LSTMCellGroup::integrate(Tensor* p_input, Tensor* p_weights)
{
	_net += *p_weights * *p_input;
}

void LSTMCellGroup::activate()
{
	activate(_input_gate->get_output(), _output_gate->get_output(), _forget_gate->get_output());
}

void LSTMCellGroup::reset() const
{
	_state.fill(0);
}

void LSTMCellGroup::activate(Tensor* p_input_gate, Tensor* p_output_gate, Tensor* p_forget_gate)
{
	if (is_bias()) {
		_net += *_bias;
	}

	float *gox = &_g_output.arr()[0];
	float *hox = &_h_output.arr()[0];
	float *sx = &_state.arr()[0];
	float *nx = &_net.arr()[0];
	float *ox = &_output.arr()[0];
	float *dix = &_deriv_input.arr()[0];

	float *igx = &p_input_gate->arr()[0];
	float *ogx = &p_output_gate->arr()[0];
	float *fgx = &p_forget_gate->arr()[0];

	_g_output = _g->activate(_net);

	for (int i = 0; i < _dim; i++)
	{
		*sx = *sx * *fgx++ + *gox++ * *igx++;
		sx++;
	}

	_h_output = _f->activate(_state);

	for (int i = 0; i < _dim; i++)
	{
		*ox++ = *hox++ * *ogx++;
		*dix++ = *nx;
		*nx++ = 0;
	}

	/*
	for (int i = 0; i < _dim; i++)
	{
		_g_output[i] = _g->activate(_net[i]);
		_state[i] = _state[i] * p_forget_gate->at(i) + _g_output[i] * p_input_gate->at(i);
		_h_output[i] = _f->activate(_state[i]);
		_output[i] = _h_output[i] * p_output_gate->at(i);
	}

	_deriv_input = _net;
	_net.fill(0);
	*/
}

Tensor LSTMCellGroup::get_h() const
{
	return _h_output;
}

Tensor LSTMCellGroup::get_dh() {
	return _f->derivative(_state);
}

Tensor LSTMCellGroup::get_g() const
{
	return _g_output;
}

Tensor LSTMCellGroup::get_dg() {
	return _g->derivative(_deriv_input);
}

json LSTMCellGroup::get_json() const
{
	json data = BaseCellGroup::get_json();

	data["f"] = _f->get_json();
	data["g"] = _g->get_json();

	return data;
}

LSTMCellGroup::LSTMCellGroup(LSTMCellGroup* p_source, SimpleCellGroup* p_input_gate, SimpleCellGroup* p_output_gate, SimpleCellGroup* p_forget_gate) : BaseCellGroup(p_source)
{
	_state = Tensor::Zero({ p_source->_dim });
	_g_output = Tensor::Zero({ p_source->_dim });
	_h_output = Tensor::Zero({ p_source->_dim });
	_output = Tensor::Zero({ p_source->_dim });

	_f = init_activation_function(p_source->_f->get_type());
	_g = init_activation_function(TANH);

	_input_gate = p_input_gate;
	_forget_gate = p_forget_gate;
	_output_gate = p_output_gate;
}
