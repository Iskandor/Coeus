#include "LSTMLayer.h"
#include "LSTMLayerGradient.h"

using namespace Coeus;


LSTMLayer::LSTMLayer(const string& p_id, int p_dim, ACTIVATION p_activation) : BaseLayer(p_id)
{
	_type = LSTM;

	_input_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, false));
	_output_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, true));
	_forget_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, false));
	_cec = add_group<LSTMCellGroup>(new LSTMCellGroup(p_dim, p_activation, _input_gate, _output_gate, _forget_gate));
	_context = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, LINEAR, false));

	_output_group = _cec;

	_ct_cec = add_connection(new Connection(p_dim, p_dim, _context->get_id(), _cec->get_id()));
	_ct_cec->init(Connection::LECUN_UNIFORM);
}

LSTMLayer::~LSTMLayer()
{
	delete _cec;
	delete _input_gate;
	delete _output_gate;
	delete _forget_gate;
	delete _aux_input;
	delete _input_group;
	delete _context;

	delete _in_input_gate;
	delete _in_output_gate;
	delete _in_forget_gate;
	delete _ct_cec;
}

void LSTMLayer::init(vector<BaseLayer*>& p_input_layers)
{
	int dim = 0;
	for (auto& layer : p_input_layers)
	{
		dim += layer->get_output()->size();
	}

	const int aux_dim = dim + _cec->get_dim();

	_input_group = add_group<SimpleCellGroup>(new SimpleCellGroup(dim, LINEAR, false));
	_aux_input = add_group<SimpleCellGroup>(new SimpleCellGroup(aux_dim, LINEAR, false));
	_in_input_gate = add_connection(new Connection(_aux_input->get_dim(), _input_gate->get_dim(), _aux_input->get_id(), _input_gate->get_id()));
	_in_input_gate->init(Connection::LECUN_UNIFORM);
	_in_output_gate = add_connection(new Connection(_aux_input->get_dim(), _output_gate->get_dim(), _aux_input->get_id(), _output_gate->get_id()));
	_in_output_gate->init(Connection::LECUN_UNIFORM);
	_in_forget_gate = add_connection(new Connection(_aux_input->get_dim(), _forget_gate->get_dim(), _aux_input->get_id(), _forget_gate->get_id()));
	_in_forget_gate->init(Connection::LECUN_UNIFORM);

	_partial_deriv[_in_input_gate->get_id()] = Tensor::Zero({ _cec->get_dim(), aux_dim });
	_partial_deriv[_in_forget_gate->get_id()] = Tensor::Zero({ _cec->get_dim(), aux_dim });
	_partial_deriv[_input_group->get_id() + " " + _cec->get_id()] = Tensor::Zero({ _cec->get_dim(), dim });
	_partial_deriv[_ct_cec->get_id()] = Tensor::Zero({ _cec->get_dim(), _cec->get_dim() });

	_partial_deriv[_cec->get_id()] = Tensor::Zero({ _cec->get_dim() });
	_partial_deriv[_input_gate->get_id()] = Tensor::Zero({ _cec->get_dim() });
	_partial_deriv[_forget_gate->get_id()] = Tensor::Zero({ _cec->get_dim() });
}

void LSTMLayer::integrate(Tensor* p_input, Tensor* p_weights)
{
	_cec->integrate(p_input, p_weights);
	_input.emplace_back(p_input);
}

void LSTMLayer::activate(Tensor* p_input)
{
	_input_group->set_output(_input);
	_context->set_output(_cec->get_output());

	_input.emplace_back(_cec->get_output());
	_aux_input->set_output(_input);

	_input_gate->integrate(_aux_input->get_output(), _in_input_gate->get_weights());
	_input_gate->activate();
	_output_gate->integrate(_aux_input->get_output(), _in_output_gate->get_weights());
	_output_gate->activate();
	_forget_gate->integrate(_aux_input->get_output(), _in_forget_gate->get_weights());
	_forget_gate->activate();

	_cec->integrate(_context->get_output(), _ct_cec->get_weights());
	_cec->activate();

	const Tensor g = _cec->get_g();
	const Tensor h = _cec->get_h();
	const Tensor dg = _cec->get_dg();

	double d0;
	double d1;

	for(int j = 0; j < _cec->get_dim(); j++)
	{
		for(int m = 0; m < _aux_input->get_dim(); m++)
		{
			d0 = _partial_deriv[_in_input_gate->get_id()].at(j, m) * _forget_gate->get_output()->at(j);
			d1 = g[j] * _input_gate->get_deriv_output()->at(j,j) * _aux_input->get_output()->at(m);
			_partial_deriv[_in_input_gate->get_id()].set(j, m, d0 + d1);

			d0 = _partial_deriv[_in_forget_gate->get_id()].at(j, m) * _forget_gate->get_output()->at(j);
			d1 = h[j] * _forget_gate->get_deriv_output()->at(j,j) * _aux_input->get_output()->at(m);
			_partial_deriv[_in_forget_gate->get_id()].set(j, m, d0 + d1);
		}

		for (int m = 0; m < _input_group->get_dim(); m++)
		{
			d0 = _partial_deriv[_input_group->get_id() + " " + _cec->get_id()].at(j, m) * _forget_gate->get_output()->at(j);
			d1 = dg[j] * _input_gate->get_output()->at(j) * _input_group->get_output()->at(m);
			_partial_deriv[_input_group->get_id() + " " + _cec->get_id()].set(j, m, d0 + d1);
		}

		for (int m = 0; m < _context->get_dim(); m++)
		{
			d0 = _partial_deriv[_ct_cec->get_id()].at(j, m) * _forget_gate->get_output()->at(j);
			d1 = dg[j] * _input_gate->get_output()->at(j) * _context->get_output()->at(m);
			_partial_deriv[_ct_cec->get_id()].set(j, m, d0 + d1);
		}

		d0 = _partial_deriv[_input_gate->get_id()][j] * _forget_gate->get_output()->at(j);
		d1 = g[j] * _input_gate->get_deriv_output()->at(j, j);
		_partial_deriv[_input_gate->get_id()][j] = d0 + d1;

		d0 = _partial_deriv[_forget_gate->get_id()][j] * _forget_gate->get_output()->at(j);
		d1 = g[j] * _forget_gate->get_deriv_output()->at(j, j);
		_partial_deriv[_forget_gate->get_id()][j] = d0 + d1;

		d0 = _partial_deriv[_cec->get_id()].at(j) * _forget_gate->get_output()->at(j);
		d1 = dg[j] * _input_gate->get_output()->at(j);
		_partial_deriv[_cec->get_id()].set(j, d0 + d1);
	}

	_input.clear();
}

void LSTMLayer::override(BaseLayer* p_source)
{
//#TODO doplnit prepis parametrov LSTM siete
}

void LSTMLayer::reset()
{
	_partial_deriv[_in_input_gate->get_id()].fill(0);
	_partial_deriv[_in_forget_gate->get_id()].fill(0);
	_partial_deriv[_input_group->get_id() + " " + _cec->get_id()].fill(0);
	_partial_deriv[_ct_cec->get_id()].fill(0);

	_partial_deriv[_input_gate->get_id()].fill(0);
	_partial_deriv[_forget_gate->get_id()].fill(0);
	_partial_deriv[_cec->get_id()].fill(0);

	_cec->reset();
}
