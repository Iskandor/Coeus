#include "Connection.h"
#include "RandomGenerator.h"

using namespace Coeus;

Connection::Connection(const int p_in_dim, const int p_out_dim, const string& p_in_id, const string& p_out_id, bool p_trainable)
{
	_id = p_out_id + "_" + p_in_id;
	_in_dim = p_in_dim;
	_out_dim = p_out_dim;
	_in_id = p_in_id;
	_out_id = p_out_id;
	_trainable = p_trainable;
	_norm = Tensor::Zero({_out_dim});
	_weights = nullptr;
}

Connection::Connection(const int p_in_dim, const int p_out_dim, const string& p_in_id, const string& p_out_id, INIT p_init, bool p_trainable, float p_arg1, float p_arg2)
{
    _id = p_out_id + "_" + p_in_id;
	_in_dim = p_in_dim;
    _out_dim = p_out_dim;
	_in_id = p_in_id;
	_out_id = p_out_id;
	_trainable = p_trainable;
	_norm = Tensor::Zero({ _out_dim });
	init(p_init, p_trainable, p_arg1, p_arg2);
}

Connection::Connection(json p_data) {
	_id = p_data["id"].get<string>();
	_in_id = p_data["in_id"].get<string>();
	_out_id = p_data["out_id"].get<string>();
	_in_dim = p_data["in_dim"].get<int>();
	_out_dim = p_data["out_dim"].get<int>();
	_trainable = p_data["trainable"].get<bool>();

	float* data = Tensor::alloc_arr(_out_dim * _in_dim);

	stringstream ss(p_data["weights"].get<string>());

	ss.seekg(0,ios::end);
	const streampos size = ss.tellg();
	ss.seekg(0, ios::beg);
	ss.read(reinterpret_cast<char*>(data), size);

	_weights = new Tensor({_out_dim, _in_dim}, data);

	if (_trainable)
	{
		add_param(_id, _weights);
	}
}

Connection* Connection::clone() const
{
	Connection* result = new Connection(_in_dim, _out_dim, _in_id, _out_id, _trainable);
	result->_weights = _weights;

	if (result->_trainable)
	{
		result->add_param(result->_id, result->_weights);
	}

	return result;
}

Connection::~Connection()
= default;

void Connection::init(const INIT p_init, const bool p_trainable, const float p_arg1, const float p_arg2) {
    switch(p_init) {
		case NONE:			
			_weights = new Tensor({ _out_dim, _in_dim }, Tensor::ZERO);
			break;
        case UNIFORM:
            uniform(p_arg1, p_arg2);
            break;
        case LECUN_UNIFORM:
            uniform(-pow(_in_dim, -.5f), pow(_in_dim, -.5f));
            break;
        case GLOROT_UNIFORM:
            uniform(-2.f / (_in_dim + _out_dim), 2.f / (_in_dim + _out_dim));
            break;
        case IDENTITY:
            identity();
            break;
	    case NORMAL:
			normal(p_arg1, p_arg2);
    		break;
	    case EXPONENTIAL:
			exponential(p_arg1);
    		break;
	    case HE_UNIFORM:
			uniform(-sqrt(6.f / _in_dim), sqrt(6.f / _in_dim));
    		break;
	    case LECUN_NORMAL:
			normal(0., sqrt(1.f / _in_dim));
    		break;		
	    case GLOROT_NORMAL:
			normal(0., 2.f / (_in_dim + _out_dim));
    		break;
	    case HE_NORMAL:
			normal(0., sqrt(2.f / _in_dim));
    		break;
    }
	_trainable = p_trainable;

	if (p_trainable)
	{
		add_param(_id, _weights);
	}
}

json Connection::get_json() const
{
	json result;

	result["id"] = _id;
	result["in_id"] = _in_id;
	result["out_id"] = _out_id;
	result["in_dim"] = _in_dim;
	result["out_dim"] = _out_dim;
	result["trainable"] = _trainable;

	stringstream ss;

	for (int i = 0; i < _weights->size(); i++) {
		float w = (*_weights)[i];
		ss.write((char*)&w, sizeof(float));
	}

	result["weights"] = ss.str();

	return result;
}

void Connection::uniform(const float p_min, const float p_max) {
	_weights = new Tensor({ _out_dim, _in_dim }, Tensor::ZERO);

	for (int i = 0; i < _weights->size(); i++)
	{
		(*_weights)[i] = RandomGenerator::get_instance().random(p_min, p_max);
	}
}

void Connection::normal(const float p_mean, const float p_dev)
{
	_weights = new Tensor({ _out_dim, _in_dim }, Tensor::ZERO);

	for(int i = 0; i < _weights->size(); i++)
	{
		(*_weights)[i] = RandomGenerator::get_instance().normal_random(p_mean, p_dev);
	}
}

void Connection::exponential(const float p_lambda)
{
	_weights = new Tensor({ _out_dim, _in_dim }, Tensor::ZERO);

	for (int i = 0; i < _weights->size(); i++)
	{
		(*_weights)[i] = RandomGenerator::get_instance().exp_random(p_lambda);
	}
}

void Connection::identity() {
	_weights = new Tensor({ _out_dim, _in_dim }, Tensor::ONES);
}

void Connection::set_weights(Tensor *p_weights) const {
    _weights->override(p_weights);
}

void Connection::update_weights(Tensor& p_delta_w) const
{
	*_weights += p_delta_w;
}

void Connection::normalize_weights(const NORM p_norm) const {

	switch (p_norm) {
	case L1_NORM:
		
		for (int i = 0; i < _out_dim; i++) {
			_norm[i] = 0;
			for (int j = 0; j < _in_dim; j++) {
				_norm[i] += abs(_weights->at(i, j));
			}
		}

		break;
	case L2_NORM:
		for (int i = 0; i < _out_dim; i++) {
			_norm[i] = 0;
			for (int j = 0; j < _in_dim; j++) {
				_norm[i] += pow(_weights->at(i, j), 2);
			}
			_norm[i] = sqrt(_norm[i]);
		}

		break;

	}

	for (int i = 0; i < _out_dim; i++) {
		for (int j = 0; j < _in_dim; j++) {
			if (_norm[i] > 0) {
				_weights->set(i, j, _weights->at(i, j) / _norm[i]);
			}			
		}
	}
}

void Connection::override(Connection* p_copy) {
	_weights->override(p_copy->_weights);
	_trainable = p_copy->_trainable;
}
