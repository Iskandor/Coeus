#include "Connection.h"

using namespace Coeus;

Connection::Connection(const int p_in_dim, const int p_out_dim, const string& p_in_id, const string& p_out_id)
{
    _id = p_out_id + "_" + p_in_id;
	_in_dim = p_in_dim;
    _out_dim = p_out_dim;
	_in_id = p_in_id;
	_out_id = p_out_id;
	_trainable = true;

	_norm = Tensor::Zero({ _out_dim });
}

Connection::Connection(json p_data) {
	_id = p_data["id"].get<string>();
	_in_id = p_data["in_id"].get<string>();
	_out_id = p_data["out_id"].get<string>();
	_in_dim = p_data["in_dim"].get<int>();
	_out_dim = p_data["out_dim"].get<int>();
	_trainable = p_data["trainable"].get<bool>();

	double* data = Tensor::alloc_arr(_out_dim * _in_dim);

	stringstream ss(p_data["weights"].get<string>());

	ss.seekg(0,ios::end);
	const streampos size = ss.tellg();
	ss.seekg(0, ios::beg);
	ss.read(reinterpret_cast<char*>(data), size);

	_weights = Tensor({_out_dim, _in_dim}, data);
}

Connection::~Connection()
{
}

void Connection::init(const INIT p_init, const bool p_trainable, const double p_limit) {
    switch(p_init) {
		case NONE:			
			_weights = Tensor::Zero({ _out_dim, _in_dim });
			break;
        case UNIFORM:
            uniform(p_limit);
            break;
        case LECUN_UNIFORM:
            uniform(static_cast<double>(pow(_in_dim, -.5)));
            break;
        case GLOROT_UNIFORM:
            uniform(2.0f / (_in_dim + _out_dim));
            break;
        case IDENTITY:
            identity();
            break;
    }
	_trainable = p_trainable;
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

	for (int i = 0; i < _weights.size(); i++) {
		double w = _weights[i];
		ss.write((char*)&w, sizeof(double));
	}

	result["weights"] = ss.str();

	return result;
}

void Connection::uniform(const double p_limit) {
	_weights = Tensor({ _out_dim, _in_dim }, Tensor::RANDOM, p_limit);
}

void Connection::identity() {
	_weights = Tensor::Ones({ _out_dim, _in_dim });
}

void Connection::set_weights(Tensor *p_weights) const {
    _weights.override(p_weights);
}

void Connection::update_weights(Tensor& p_delta_w) {
	_weights += p_delta_w;
}

void Connection::normalize_weights(const NORM p_norm) const {

	switch (p_norm) {
	case L1_NORM:
		
		for (int i = 0; i < _out_dim; i++) {
			_norm[i] = 0;
			for (int j = 0; j < _in_dim; j++) {
				_norm[i] += abs(_weights.at(i, j));
			}
		}

		break;
	case L2_NORM:
		for (int i = 0; i < _out_dim; i++) {
			_norm[i] = 0;
			for (int j = 0; j < _in_dim; j++) {
				_norm[i] += pow(_weights.at(i, j), 2);
			}
			_norm[i] = sqrt(_norm[i]);
		}

		break;

	}

	for (int i = 0; i < _out_dim; i++) {
		for (int j = 0; j < _in_dim; j++) {
			if (_norm[i] > 0) {
				_weights.set(i, j, _weights.at(i, j) / _norm[i]);
			}			
		}
	}
}

void Connection::override(Connection* p_copy) {
	_weights.override(&p_copy->_weights);
	_trainable = p_copy->_trainable;
}
